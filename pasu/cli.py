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

from PIL import Image
from tctim import imprint
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from torchvision.transforms import Pad, ToTensor, Compose, Normalize, RandomHorizontalFlip, RandomCrop, RandomResizedCrop
from torch import nn

from ntorx.attribution import DTDZPlus, DTDZB, ShapeAttributor, SequentialAttributor, PassthroughAttributor, GradientAttributor
from ntorx.image import colorize, montage, imgify
from ntorx.nn import Linear, PaSU
from ntorx.util import config_logger

from .model import FeedFwd, VGG16


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
@click.option('--nout', type=int, default=10)
@click.option('--cin', type=int, default=3)
@click.option('--beta', type=float, default=1e2)
@click.option('--force-relu/--no-force-relu', default=False)
@click.option('--batchnorm/--no-batchnorm', default=False)
@click.option('--parallel/--no-parallel', default=False)
@click.pass_context
def model(ctx, load, compat, nout, cin, beta, force_relu, batchnorm, parallel):
    #model = FeedFwd((1, 32, 32), 10, relu=force_relu, beta=beta)
    init_weights = (load is None) or (compat is not None)
    model = GradientAttributor.of(VGG16)(cin, nout, relu=force_relu, beta=beta, init_weights=init_weights, batch_norm=batchnorm, parallel=parallel)
    if load is not None:
        if compat is None:
            model.load_params(load)
        else:
            trans = json.load(compat)
            model.load_params(load, trans=trans, strict=False)

    model.device(ctx.obj.device)
    ctx.obj.model = model

@main.command()
@click.option('-b', '--bsize', type=int, default=32)
@click.option('--train/--test', default=True)
@click.option('--datapath', default=path.join(xdg_data_home(), 'dataset'))
@click.option('--dataset', type=click.Choice(['CIFAR10', 'MNIST', 'Imagenet-12']), default='CIFAR10')
@click.option('--download/--no-download', default=False)
@click.option('--shuffle/--no-shuffle', default=True)
@click.option('--workers', type=int, default=4)
@click.pass_context
def data(ctx, bsize, train, datapath, dataset, download, shuffle, workers):
    if dataset == 'CIFAR10':
        transf = Compose(([RandomCrop(32, padding=4), RandomHorizontalFlip()] if train else []) + [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dset = CIFAR10(root=datapath, train=train , transform=transf, download=download)
    elif dataset == 'MNIST':
        dset = MNIST(root=data, train=train , transform=Compose([Pad(2), ToTensor()]), download=download)
    elif dataset == 'Imagenet-12':
        transf = Compose(([RandomResizedCrop(224), RandomHorizontalFlip()] if train else [CenterCrop(224)]) + [ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        dset = ImageFolder(root=datapath, transform=transf)
        if not train:
            logger.warning('Imagenet-12 requires a different data path for validation!')
    else:
        raise RuntimeError('No such dataset!')
    loader  = DataLoader(dset, bsize, shuffle=shuffle, num_workers=workers)
    ctx.obj.loader = loader

@main.command()
@click.option('--param', type=ChoiceList(['weight', 'bias', 'beta', 'all']), default=['weight', 'bias'])
@click.option('--lr', type=float, default=1e-3)
@click.option('--decay', type=float)
@click.pass_context
def optimize(ctx, param, lr, decay):
    model = ctx.obj.model

    optargs = []
    if 'weight' in param:
        params = (module.weight for module in model.modules() if isinstance(module, Linear))
        arg = {'params': params, 'lr': lr}
        #if decay:
        #    arg['weight_decay'] = decay
        optargs.append(arg)
    if 'bias' in param:
        params = (module.bias for module in model.modules() if isinstance(module, Linear))
        arg = {'params': params, 'lr': lr}
        #if decay:
        #    arg['weight_decay'] = decay
        optargs.append(arg)
    if 'beta' in param:
        params = (module.beta for module in model.modules() if isinstance(module, PaSU))
        arg = {'params': params, 'lr': lr}
        if decay:
            arg['weight_decay'] = decay
        optargs.append(arg)
    if 'all' in param:
        arg = {'params': model.parameters(), 'lr': lr}
        #if decay:
        #    arg['weight_decay'] = decay
        optargs.append(arg)

    ctx.obj.optimizer = torch.optim.Adam(optargs)

@main.command()
@click.option('-c', '--checkpoint', type=click.Path())
@click.option('-s', '--start', type=int, default=0)
@click.option('-n', '--nepochs', type=int, default=10)
@click.option('-f', '--sfreq', type=int, default=1)
@click.option('--nslope', type=int, default=5)
@click.pass_context
def train(ctx, checkpoint, start, nepochs, sfreq, nslope):
    loader = ctx.obj.loader
    model = ctx.obj.model
    optimizer = ctx.obj.optimizer

    model.train_params(loader, optimizer, nepochs=nepochs, nslope=nslope, spath=checkpoint, start=start, sfreq=sfreq)

@main.command()
@click.option('-o', '--output', type=click.File(mode='w'), default=stdout)
@click.pass_context
def validate(ctx, output):
    loader = ctx.obj.loader
    model = ctx.obj.model

    acc = model.test_params(loader)
    output.write('Accuracy: {:.3f}\n'.format(acc))

@main.command()
@click.option('-o', '--output', type=click.File(mode='w'), default=stdout)
@click.pass_context
def betastat(ctx, output):
    model = ctx.obj.model

    params = (module.beta for module in model.modules() if isinstance(module, PaSU))
    stats = [(param.data.mean().item(), param.data.std().item()) for param in params]

    head = '{: ^3} {: ^10} {: ^10}\n'.format('#', 'mean', 'std')
    info = '\n'.join('{: >3d} {: >10.2e} {: >10.2e}'.format(n, *stat) for n, stat in enumerate(stats)) + '\n'
    output.write(head + info)

@main.command()
@click.option('-o', '--output', type=click.File(mode='wb'), default=stdout.buffer)
@click.option('-b', '--backup', type=click.File(mode='wb'))
@click.pass_context
def attribution(ctx, output, backup):
    loader = ctx.obj.loader
    model = ctx.obj.model
    model.to(ctx.obj.device)
    model.eval()

    data, label = next(iter(loader))
    data = data.to(ctx.obj.device)
    #attrib = model.attribution(model(data))
    attrib = model.attribution(inpt=data)

    #carr = np.moveaxis(attrib.detach().cpu().numpy(), 1, -1)
    carr = np.abs(np.moveaxis(attrib.detach().cpu().numpy(), 1, -1)).sum(-1, keepdims=True)
    carr /= np.abs(carr).sum((1, 2, 3), keepdims=True)
    if output.isatty():
        imprint(colorize(carr.squeeze(3), cmap='bwr'), montage=True)
    else:
        img = colorize(montage(carr).squeeze(2), cmap='wred')
        Image.fromarray(imgify(img)).save(output, format='png')

    if backup:
        barr = montage(np.moveaxis(data.detach().cpu().numpy(), 1, -1))
        Image.fromarray(imgify(barr)).save(backup, format='png')

if __name__ == '__main__':
    main(auto_envvar_prefix='PASU')
