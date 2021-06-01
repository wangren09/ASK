import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        cfg1 = [64, 64, 'M']
        cfg2 = [128, 128, 'M']
        cfg3 = [256, 256, 256, 'M']
        cfg4 = [512, 512, 512, 'M']
        cfg5 = [512, 512, 512, 'M']
        self.layer1 = self._make_layers(cfg1, 3)
        self.layer2 = self._make_layers(cfg2, 64)
        self.layer3 = self._make_layers(cfg3, 128)
        self.layer4 = self._make_layers(cfg4, 256)
        self.layer5 = self._make_layers(cfg5, 512)
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

    def _make_layers(self, cfg, in_channels):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def forward(self, x, extra_out=None):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out = self.classifier(out5)
        if extra_out is None:
            return out
        else:
            return [out1, out2, out3, out4, out5][extra_out].flatten(start_dim=1), out
