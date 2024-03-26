import torch.nn as nn
import torchvision.models as models
import torch

class Resnet(nn.Module):
    def __init__(self, output_dims=64, channel=2, pretrained=True, norm=False):
        super().__init__()
        self.model=models.resnet18(pretrained)
        self.model.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, output_dims)
        self.norm=norm
        self.batch_norm=nn.BatchNorm2d(1)
        # self.dropout=nn.Dropout(p=0.3)

    def forward(self,x):
        # x=self.dropout(x)
        # x=self.batch_norm(x)
        # x=self.dropout(x)
        if self.norm:
            mean=torch.mean(x,dim=-2,keepdim=True)
            std=torch.std(x,dim=-2,keepdim=True)
            y=(x-mean)/(std+1e-8)
        else:
            y=x
        # y=self.dropout(y)
        y=self.batch_norm(y)
        # y=self.dropout(y)
        return self.model(y)

class Linear(nn.Module):
    def __init__(self, input_dims=64, output_dims=6):
        super().__init__()
        self.model=nn.Sequential(
            # nn.Dropout(p=0.3),
            # nn.BatchNorm1d(64),
            nn.Linear(input_dims,64),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            # nn.BatchNorm1d(64),
            nn.Linear(64,64),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            # nn.BatchNorm1d(64),
            nn.Linear(64, output_dims)
        )

    def forward(self,x):
        return self.model(x)