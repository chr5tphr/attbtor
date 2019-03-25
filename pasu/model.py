import sys

from collections import OrderedDict

from torch.nn import init
from torch.nn import Dropout, MaxPool2d, AdaptiveAvgPool2d
from torchvision.models.vgg import cfg as vggconfig
from ntorx.attribution import DTDZPlus, DTDZB, ShapeAttributor, SequentialAttributor, PassthroughAttributor, GradientAttributor
from ntorx.model import Parametric, FeedForwardParametric
from ntorx.nn import BatchView, PaSU, Sequential, Linear
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

class VGG16(FeedForwardParametric):
    def __init__(self, in_dim, out_dim, relu=True, beta=20, init_weights=True, batch_norm=False, **kwargs):
        super().__init__(**kwargs)
        def make_layers(cfg, batch_norm=False):
            layers = OrderedDict()
            in_channels = in_dim
            for n, v in enumerate(cfg):
                if v == 'M':
                    layers.update([
                        ('pool%d'%n, MaxPool2d(kernel_size=2, stride=2)),
                    ])
                else:
                    conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
                    if batch_norm:
                        layers.update([
                            ('conv%d'%n, conv2d),
                            ('bnrm%d'%n, BatchNorm2d(v)),
                            ('pasu%d'%n, PaSU(v, relu=relu, init=beta))
                        ])
                    else:
                        layers.update([
                            ('conv%d'%n, conv2d),
                            ('pasu%d'%n, PaSU(v, relu=relu, init=beta))
                        ])
                    in_channels = v
            return Sequential(layers)
        self.features = make_layers(vggconfig['D'], batch_norm=batch_norm)
        self.avgpool = AdaptiveAvgPool2d((7, 7))
        self.classifier = Sequential(OrderedDict([
            ('dens0', Dense(512 * 7 * 7, 4096)),
            ('pasu0', PaSU(4096, relu=relu, init=beta)),
            ('drop0', Dropout()),
            ('dens1', Dense(4096, 4096)),
            ('pasu1', PaSU(4096, relu=relu, init=beta)),
            ('drop1', Dropout()),
            ('dens2', Dense(4096, out_dim)),
        ]))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, Dense):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)

