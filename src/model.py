import torch.nn as nn
from monai.networks.nets import resnet10, resnet34, resnet50, EfficientNetBN


class AclNet(nn.Module):
    def __init__(self, architecture):
        super(AclNet, self).__init__()

        if architecture.startswith('resnet'):
            net = build_resnet(architecture)
        elif architecture.startswith('efficientnet'):
            net = build_efficientnet(architecture)

        self.model = net

    def forward(self, x):
        return self.model(x)


def build_resnet(architecture):
    return {
        'resnet10': resnet10,
        'resnet34': resnet34,
        'resnet50': resnet50,
    }[architecture](
        pretrained=False,
        spatial_dims=3,
        n_input_channels=1,
        num_classes=1
    )


def build_efficientnet(architecture):
    return EfficientNetBN(architecture, pretrained=False, spatial_dims=3, in_channels=1, num_classes=1)
