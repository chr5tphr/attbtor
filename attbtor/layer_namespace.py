from functools import partialmethod
from argparse import Namespace

from torch.nn import init, Dropout, MaxPool2d, AdaptiveAvgPool2d, ReLU
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

class PartialType(type):
    def __new__(cls, base, *args, **kwargs):
        return type(base.__name__, (base,), {'__init__': partialmethod(base.__init__, *args, **kwargs)})

preset = {
    'None': Namespace(
        Dense             = Dense,
        Conv2d            = Conv2d,
        IConv2d           = Conv2d,
        ReLU              = ReLU,
        Dropout           = Dropout,
        MaxPool2d         = MaxPool2d,
        AdaptiveAvgPool2d = AdaptiveAvgPool2d,
        BatchView         = BatchView,
        Sequential        = Sequential,
    ),
    'LRPSeqA': Namespace(
        Dense             = PartialType(LRPEpsilon.of(Dense), use_bias=True),
        Conv2d            = PartialType(DTDZPlus.of(Conv2d), use_bias=True),
        IConv2d           = LRPFlat.of(Conv2d),
        ReLU              = PassthroughAttributor.of(ReLU),
        Dropout           = PassthroughAttributor.of(Dropout),
        MaxPool2d         = PoolingAttributor.of(MaxPool2d),
        AdaptiveAvgPool2d = PoolingAttributor.of(AdaptiveAvgPool2d),
        BatchView         = ShapeAttributor.of(BatchView),
        Sequential        = SequentialAttributor,
    ),
    'LRPSeqB': Namespace(
        Dense             = PartialType(LRPEpsilon.of(Dense), use_bias=True),
        Conv2d            = PartialType(LRPAlphaBeta.of(Conv2d), use_bias=True, alpha=2, beta=1),
        IConv2d           = LRPFlat.of(Conv2d),
        ReLU              = PassthroughAttributor.of(ReLU),
        Dropout           = PassthroughAttributor.of(Dropout),
        MaxPool2d         = PoolingAttributor.of(MaxPool2d),
        AdaptiveAvgPool2d = PartialType(
            PoolingAttributor.of(AdaptiveAvgPool2d),
            pool_op=(lambda x: avg_pool2d(x, kernel_size=2, stride=2))
        ),
        BatchView         = ShapeAttributor.of(BatchView),
        Sequential        = SequentialAttributor,
    ),
    'DTD': Namespace(
        Dense             = PartialType(DTDZPlus.of(Dense), use_bias=False),
        Conv2d            = PartialType(DTDZPlus.of(Conv2d), use_bias=False),
        IConv2d           = PartialType(DTDZB.of(Conv2d), use_bias=False, lo=-5., hi=5.),
        ReLU              = PassthroughAttributor.of(ReLU),
        Dropout           = PassthroughAttributor.of(Dropout),
        MaxPool2d         = PoolingAttributor.of(MaxPool2d),
        AdaptiveAvgPool2d = PoolingAttributor.of(AdaptiveAvgPool2d),
        kwpool            = {},
        BatchView         = ShapeAttributor.of(BatchView),
        Sequential        = SequentialAttributor,
    ),
}
