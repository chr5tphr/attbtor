import sys
import json

from argparse import Namespace
from collections import OrderedDict
from itertools import chain
from logging import getLogger
from os import path, environ
from sys import stdout

import click
import numpy as np
import torch
import h5py
import re

from PIL import Image
from tctim import imprint
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from torchvision.transforms import Pad, ToTensor, Compose, Normalize, RandomHorizontalFlip, RandomCrop, RandomResizedCrop, CenterCrop
from torch import nn

from ntorx.attribution import DTDZPlus, DTDZB, ShapeAttributor, SequentialAttributor, PassthroughAttributor, GradientAttributor, SmoothGradAttributor
from ntorx.image import colorize, montage, imgify
from ntorx.nn import Linear, PaSU
from ntorx.util import config_logger

from .model import FeedFwd, VGG16, preset


logger = getLogger()

def xdg_data_home():
    return environ.get('XDG_DATA_HOME', path.join(environ['HOME'], '.local', 'share'))

class ChoiceList(click.Choice):
    def __init__(self, *args, separator=',', **kwargs):
        super().__init__(*args, **kwargs)
        self.separator = separator

    def convert(self, values, param, ctx):
        retval = []
        for val in values.split(self.separator):
            retval.append(super(click.Choice, self).convert(val, param, ctx))
        return retval


@click.group(chain=True)
@click.option('--log', type=click.File(), default=stdout)
@click.option('-v', '--verbose', count=True)
@click.option('--threads', type=int, default=0)
@click.option('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
@click.pass_context
def main(ctx, log, verbose, threads, device):
    torch.set_num_threads(threads)
    config_logger(log, level='DEBUG' if verbose > 0 else 'INFO')

    ctx.ensure_object(Namespace)
    ctx.obj.device = torch.device(device)

@main.command()
@click.option('-l', '--load', type=click.Path())
@click.option('--compat', type=click.File())
@click.option('--nout', type=int, default=1000)
@click.option('--cin', type=int, default=3)
@click.option('--attributor', type=click.Choice(['gradient', 'smoothgrad', 'dtd', 'lrp_a', 'lrp_b']), default='gradient')
@click.option('--parallel/--no-parallel', default=False)
@click.pass_context
def model(ctx, load, compat, nout, cin, attributor, parallel):
    init_weights = (load is None) or (compat is not None)
    Model = {
        'gradient': GradientAttributor,
        'smoothgrad': SmoothGradAttributor,
        'dtd': SequentialAttributor,
        'lrp_a': SequentialAttributor,
        'lrp_b': SequentialAttributor,
    }[attributor].of(VGG16)
    layerns = preset[{
        'gradient': 'None',
        'smoothgrad': 'None',
        'dtd': 'DTD',
        'lrp_a': 'LRPSeqA',
        'lrp_b': 'LRPSeqB',
    }[attributor]]
    model = Model(cin, nout, init_weights=init_weights, parallel=parallel, layerns=layerns)
    if load is not None:
        if compat is None:
            model.load_params(load)
        else:
            trans = json.load(compat)
            model.load_params(load, trans=trans, strict=False)

    model.device(ctx.obj.device)
    ctx.obj.model = model
    ctx.obj.nout = nout

@main.command()
@click.option('-b', '--bsize', type=int, default=32)
@click.option('--train/--test', default=False)
@click.option('--datapath', default=path.join(xdg_data_home(), 'dataset'))
@click.option('--regex')
@click.option('--dataset', type=click.Choice(['CIFAR10', 'MNIST', 'Imagenet-12']), default='Imagenet-12')
@click.option('--download/--no-download', default=False)
@click.option('--shuffle/--no-shuffle', default=False)
@click.option('--workers', type=int, default=4)
@click.pass_context
def data(ctx, bsize, train, datapath, regex, dataset, download, shuffle, workers):
    if regex is not None:
        rvalid = re.compile(regex)
        def is_valid_file(fpath):
            return rvalid.fullmatch(fpath) is not None
    else:
        is_valid_file = None

    if dataset == 'CIFAR10':
        transf = Compose(([RandomCrop(32, padding=4), RandomHorizontalFlip()] if train else []) + [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dset = CIFAR10(root=datapath, train=train , transform=transf, download=download)
    elif dataset == 'MNIST':
        dset = MNIST(root=data, train=train , transform=Compose([Pad(2), ToTensor()]), download=download)
    elif dataset == 'Imagenet-12':
        transf = Compose(([RandomResizedCrop(224), RandomHorizontalFlip()] if train else [CenterCrop(224)]) + [ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        dset = ImageFolder(root=datapath, transform=transf, is_valid_file=is_valid_file)
    else:
        raise RuntimeError('No such dataset!')
    loader  = DataLoader(dset, bsize, shuffle=shuffle, num_workers=workers)
    ctx.obj.loader = loader

@main.command()
@click.option('-o', '--output', type=click.File(mode='w'), default=stdout)
@click.pass_context
def validate(ctx, output):
    loader = ctx.obj.loader
    model = ctx.obj.model

    acc = model.test_params(loader)
    output.write('Accuracy: {:.3f}\n'.format(acc))

@main.command()
@click.option('-o', '--output', type=click.Path())
@click.option('--predicted/--label', default=False)
@click.pass_context
def attribution(ctx, output, predicted):
    loader = ctx.obj.loader
    model = ctx.obj.model
    model.to(ctx.obj.device)
    model.eval()
    dlen = len(loader)
    nout = ctx.obj.nout

    for i, (data, label) in enumerate(loader):
        logger.debug('Processing batch {:d}/{:d} ...'.format(i, dlen))
        data = data.to(ctx.obj.device)
        out = model.forward(data)
        amax = out.argmax(1) if predicted else label
        fout = torch.eye(nout, device=out.device, dtype=out.dtype)[amax]
        attrib = model.attribution(fout)
        if not path.exists(output):
            subshp = tuple(attrib.shape[1:])
            with h5py.File(output, 'w') as fd:
                fd.create_dataset('attribution', shape=(0,) + subshp, dtype='float32', maxshape=(None,) + subshp, chunks=True)
                fd.create_dataset('prediction',  shape=(0,), dtype='uint16', maxshape=(None,), chunks=True)
                fd.create_dataset('label',       shape=(0,), dtype='uint16', maxshape=(None,), chunks=True)

        with h5py.File(output, 'a') as fd:
            n = fd['attribution'].shape[0]
            fd['attribution'].resize(n + attrib.shape[0], axis=0)
            fd['prediction'].resize(n + attrib.shape[0], axis=0)
            fd['label'].resize(n + attrib.shape[0], axis=0)

            fd['attribution'][n:] = attrib.detach().cpu().numpy()
            fd['prediction'][n:] = out.argmax(1).detach().cpu().numpy()
            fd['label'][n:] = label.detach().cpu().numpy()

@main.command()
@click.option('-o', '--output', type=click.File(mode='wb'), default=stdout.buffer)
@click.option('-b', '--backup', type=click.File(mode='wb'))
@click.option('--cmap', default='hot')
@click.option('--seed', type=int, default=0xDEADBEEF)
@click.pass_context
def attrvis(ctx, output, backup, cmap, seed):
    loader = ctx.obj.loader
    model = ctx.obj.model
    nout = ctx.obj.nout
    model.to(ctx.obj.device)
    model.eval()

    torch.manual_seed(seed)
    data, label = next(iter(loader))
    data = data.to(ctx.obj.device)
    out = model(data)
    amax = out.argmax(1)
    fout = torch.eye(nout, device=out.device, dtype=out.dtype)[amax]
    attrib = model.attribution(fout)

    #carr = np.abs(np.moveaxis(attrib.detach().cpu().numpy(), 1, -1)).sum(-1, keepdims=True)
    carr = np.moveaxis(attrib.detach().cpu().numpy(), 1, -1).sum(-1, keepdims=True)
    bbox = (lambda x: (-x, x))(np.abs(carr).max())
    carr /= np.abs(carr).sum((1, 2, 3), keepdims=True)
    if output.isatty():
        imprint(colorize(carr.squeeze(3), cmap=cmap, bbox=bbox), montage=True)
    else:
        img = colorize(montage(carr).squeeze(2), cmap=cmap, bbox=bbox)
        Image.fromarray(imgify(img)).save(output, format='png')

    if backup:
        barr = montage(np.moveaxis(data.detach().cpu().numpy(), 1, -1))
        Image.fromarray(imgify(barr)).save(backup, format='png')

if __name__ == '__main__':
    main(auto_envvar_prefix='PASU')
