import torch.nn as nn
from monai.networks.nets import resnet10, resnet34, resnet50


class AclNet(nn.Module):
    def __init__(self, architecture):
        super(AclNet, self).__init__()

        net = get_architecture(architecture)(
            pretrained=False,
            spatial_dims=3,
            n_input_channels=1,
            num_classes=1
        )
        self.model = net

    def forward(self, x):
        return self.model(x)


def get_architecture(x):
    return {
        'resnet10': resnet10,
        'resnet34': resnet34,
        'resnet50': resnet50,
    }[x]
