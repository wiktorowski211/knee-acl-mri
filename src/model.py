import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DenseNet121, resnet10, resnet34, resnet50


class AclNet(nn.Module):
    def __init__(self, architecture):
        super(AclNet, self).__init__()

        self.class_net = DenseNet121(spatial_dims=2, in_channels=1,
                                     out_channels=3, block_config=(1, 2, 1), dropout_prob=0.1)

        self.loc_net = DenseNet121(spatial_dims=3, in_channels=1,
                                   out_channels=12, block_config=(1, 2, 1), dropout_prob=0.1)

        torch.nn.init.uniform_(self.loc_net.class_layers.out.weight, a=-0.01, b=0.01)
        self.loc_net.class_layers.out.bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        theta = self.loc_net(x)
        theta = theta.view(-1, 3, 4)

        grid = F.affine_grid(
            theta, (x.size(0), 1, 1, 96, 96), align_corners=False)

        x = F.grid_sample(x, grid, align_corners=False)

        x = torch.squeeze(x, 2)

        sample = x[0][0]

        x = self.class_net(x)

        return x, sample


class AclNet1(nn.Module):
    def __init__(self, architecture):
        super(AclNet, self).__init__()

        net = DenseNet121(spatial_dims=3, in_channels=1, out_channels=3, block_config=(1, 1, 1, 1), dropout_prob=0.1)

        self.model = net

    def forward(self, x):
        return self.model(x)


class AclNet1(nn.Module):
    def __init__(self, architecture):
        super(AclNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(64, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))

        self.fc = nn.Sequential(
            nn.Linear(in_features=32, out_features=16),
            # nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class AclNet1(nn.Module):
    def __init__(self, architecture):
        super(AclNet, self).__init__()

        net = build_resnet(architecture)

        net.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 3),
            nn.Softmax(dim=1),
        )

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
        num_classes=3
    )
