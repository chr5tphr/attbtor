import sys

from collections import OrderedDict
from argparse import Namespace

from torch.nn import init, Dropout, MaxPool2d, AdaptiveAvgPool2d, ReLU
from torchvision.models.vgg import cfgs as vggconfig
from ntorx.attribution import LRPAlphaBeta, LRPEpsilon, PoolingAttributor, \
                              DTDZPlus, DTDZB, ShapeAttributor, SequentialAttributor, \
                              PassthroughAttributor, GradientAttributor, LRPFlat
from ntorx.model import Parametric, FeedForwardParametric
from ntorx.nn import BatchView, Sequential, Linear
from torch.nn.functional import avg_pool2d
try:
    from ntorx.nn import Dense, Conv2d, BatchNorm2d
except ImportError:
    from ntorx.nn import __getattr__ as _getlin
    for lin in ['Dense', 'Conv2d', 'BatchNorm2d']:
        setattr(sys.modules[__name__], lin, _getlin(lin))

class FeedFwd(Sequential, FeedForwardParametric):
    def __init__(self, in_dim, out_dim, relu=True, beta=1e2):
        in_flat = np.prod(in_dim)
        ABatchView = ShapeAttributor.of(BatchView)
        BDense = DTDZB.of(Dense)
        PDense = DTDZPlus.of(Dense)
        PPaSU  = PassthroughAttributor.of(PaSU)
        super().__init__(
            OrderedDict([
                ('view0', ABatchView(in_flat)),
                ('dens1', BDense(in_flat,    1024, lo=-1., hi=1.)),
                ('actv1', PPaSU(1024, relu=relu, init=beta)),
                ('dens2', PDense(   1024,    1024)),
                ('actv2', PPaSU(1024, relu=relu, init=beta)),
                ('dens3', PDense(   1024,    1024)),
                ('actv3', PPaSU(1024, relu=relu, init=beta)),
                ('dens4', PDense(   1024, out_dim)),
            ])
        )

preset = {
    'None': Namespace(
        Dense             = Dense,
        kwdense           = {},
        Conv2d            = Conv2d,
        kwconv            = {},
        IConv2d           = Conv2d,
        kwiconv           = {},
        ReLU              = ReLU,
        Dropout           = Dropout,
        MaxPool2d         = MaxPool2d,
        AdaptiveAvgPool2d = AdaptiveAvgPool2d,
        kwpool            = {},
        BatchView         = BatchView,
    ),
    'LRPSeqA': Namespace(
        Dense             = LRPEpsilon.of(Dense),
        kwdense           = {'use_bias': True},
        Conv2d            = DTDZPlus.of(Conv2d),
        kwconv            = {'use_bias': True},
        IConv2d           = LRPFlat.of(Conv2d),
        kwiconv           = {},
        ReLU              = PassthroughAttributor.of(ReLU),
        Dropout           = PassthroughAttributor.of(Dropout),
        MaxPool2d         = PoolingAttributor.of(MaxPool2d),
        AdaptiveAvgPool2d = PoolingAttributor.of(AdaptiveAvgPool2d),
        #kwpool            = {'pool_op': lambda x: avg_pool2d(x, kernel_size=2, stride=2)},
        kwpool            = {},
        BatchView         = ShapeAttributor.of(BatchView),
    ),
    'LRPSeqB': Namespace(
        Dense             = LRPEpsilon.of(Dense),
        kwdense           = {'use_bias': True},
        Conv2d            = LRPAlphaBeta.of(Conv2d),
        kwconv            = {'use_bias': True, 'alpha': 2, 'beta': 1},
        IConv2d           = LRPFlat.of(Conv2d),
        kwiconv           = {},
        ReLU              = PassthroughAttributor.of(ReLU),
        Dropout           = PassthroughAttributor.of(Dropout),
        MaxPool2d         = PoolingAttributor.of(MaxPool2d),
        AdaptiveAvgPool2d = PoolingAttributor.of(AdaptiveAvgPool2d),
        kwpool            = {'pool_op': lambda x: avg_pool2d(x, kernel_size=2, stride=2)},
        BatchView         = ShapeAttributor.of(BatchView),
    ),
    'DTD': Namespace(
        Dense             = DTDZPlus.of(Dense),
        kwdense           = {'use_bias': False},
        Conv2d            = DTDZPlus.of(Conv2d),
        kwconv            = {'use_bias': False},
        IConv2d           = DTDZB.of(Conv2d),
        kwiconv           = {'use_bias': False, 'lo': -5., 'hi': 5.},
        ReLU              = PassthroughAttributor.of(ReLU),
        Dropout           = PassthroughAttributor.of(Dropout),
        MaxPool2d         = PoolingAttributor.of(MaxPool2d),
        AdaptiveAvgPool2d = PoolingAttributor.of(AdaptiveAvgPool2d),
        #kwpool            = {'pool_op': lambda x: avg_pool2d(x, kernel_size=2, stride=2)},
        kwpool            = {},
        BatchView         = ShapeAttributor.of(BatchView),
    ),
}

class VGG16(FeedForwardParametric, Sequential):
    '''
        We can use SequentialAttributor since modules are added in order, such that _modules can simply be applied front to back
    '''
    def __init__(self, in_dim, out_dim, init_weights=False, layerns=preset['DTD'], **kwargs):
        super().__init__(**kwargs)

        R = layerns

        def make_layers(cfg):
            layers = OrderedDict()
            in_channels = in_dim
            for n, v in enumerate(cfg):
                if v == 'M':
                    layers.update([
                        ('pool%d'%n, R.MaxPool2d(kernel_size=2, stride=2, **R.kwpool)),
                    ])
                else:
                    if n == 0:
                        layers.update([
                            ('conv%d'%n, R.IConv2d(in_channels, v, kernel_size=3, padding=1, **R.kwiconv)),
                            ('relu%d'%n, R.ReLU(inplace=True))
                        ])
                    else:
                        layers.update([
                            ('conv%d'%n, R.Conv2d(in_channels, v, kernel_size=3, padding=1, **R.kwconv)),
                            ('relu%d'%n, R.ReLU(inplace=True))
                        ])
                    in_channels = v
            return SequentialAttributor(layers)
        self.features = make_layers(vggconfig['D'])
        self.avgpool = R.AdaptiveAvgPool2d((7, 7))
        self.classifier = SequentialAttributor(OrderedDict([
            ('view0', R.BatchView(-1)),
            ('dens0', R.Dense(512 * 7 * 7, 4096, **R.kwdense)),
            ('relu0', R.ReLU()),
            ('drop0', R.Dropout()),
            ('dens1', R.Dense(4096, 4096, **R.kwdense)),
            ('relu1', R.ReLU()),
            ('drop1', R.Dropout()),
            ('dens2', R.Dense(4096, out_dim, **R.kwdense)),
        ]))
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, Dense):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)

