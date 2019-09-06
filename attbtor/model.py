import sys

from collections import OrderedDict

from torchvision.models.vgg import cfgs as vggconfig

from ntorx.model import FeedForwardParametric
from ntorx.nn import Sequential

from .layer_namespace import preset

class FeedFwd(Sequential, FeedForwardParametric):
    def __init__(self, in_dim, out_dim, relu=True, beta=1e2):
        in_flat = np.prod(in_dim)
        R = preset['DTD']
        super().__init__(
            OrderedDict([
                ('view0', R.BatchView(in_flat)),
                ('dens1', R.Dense(in_flat,    1024, lo=-1., hi=1.)),
                ('actv1', R.PaSU(1024, relu=relu, init=beta)),
                ('dens2', R.Dense(   1024,    1024)),
                ('actv2', R.PaSU(1024, relu=relu, init=beta)),
                ('dens3', R.Dense(   1024,    1024)),
                ('actv3', R.PaSU(1024, relu=relu, init=beta)),
                ('dens4', R.Dense(   1024, out_dim)),
            ])
        )

class VGG(FeedForwardParametric, Sequential):
    '''
        We can use SequentialAttributor since modules are added in order, such that _modules can simply be applied front to back
    '''
    def __init__(self, in_dim, out_dim, config, init_weights=False, layerns=preset['DTD'], **kwargs):
        super().__init__(**kwargs)

        R = layerns

        def make_layers(cfg, layerns):
            layers = OrderedDict()
            in_channels = in_dim
            for n, v in enumerate(cfg):
                if v == 'M':
                    layers.update([
                        ('pool%d'%n, R.MaxPool2d(kernel_size=2, stride=2)),
                    ])
                else:
                    if n == 0:
                        layers.update([
                            ('conv%d'%n, R.IConv2d(in_channels, v, kernel_size=3, padding=1)),
                            ('relu%d'%n, R.ReLU(inplace=True))
                        ])
                    else:
                        layers.update([
                            ('conv%d'%n, R.Conv2d(in_channels, v, kernel_size=3, padding=1)),
                            ('relu%d'%n, R.ReLU(inplace=True))
                        ])
                    in_channels = v
            return R.Sequential(layers)
        self.features = make_layers(config)
        self.avgpool = R.AdaptiveAvgPool2d((7, 7))
        self.classifier = R.Sequential(OrderedDict([
            ('view0', R.BatchView(-1)),
            ('dens0', R.Dense(512 * 7 * 7, 4096)),
            ('relu0', R.ReLU()),
            ('drop0', R.Dropout()),
            ('dens1', R.Dense(4096, 4096)),
            ('relu1', R.ReLU()),
            ('drop1', R.Dropout()),
            ('dens2', R.Dense(4096, out_dim)),
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

class VGG11(VGG):
    def __init__(self, in_dim, out_dim, *args, **kwargs):
        super().__init__(in_dim, out_dim, vggconfig['A'], *args, **kwargs)

class VGG13(VGG):
    def __init__(self, in_dim, out_dim, *args, **kwargs):
        super().__init__(in_dim, out_dim, vggconfig['B'], *args, **kwargs)

class VGG16(VGG):
    def __init__(self, in_dim, out_dim, *args, **kwargs):
        super().__init__(in_dim, out_dim, vggconfig['D'], *args, **kwargs)

class VGG19(VGG):
    def __init__(self, in_dim, out_dim, *args, **kwargs):
        super().__init__(in_dim, out_dim, vggconfig['E'], *args, **kwargs)
