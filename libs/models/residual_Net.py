import torch 
import torch.nn as nn 
import torch.nn.functional as F



# _Residual Unit 
class ResidualUnit(nn.Module):
    def __init__(self, filter_in:int, filter_out:int):
        super(ResidualUnit, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_features=filter_in)
        self.conv1 = nn.Conv2d(in_channels=filter_in, out_channels=filter_out, 
                               kernel_size=3, stride=1, padding=1
        )

        self.bn2 = nn.BatchNorm2d(num_features=1)
        self.conv2 = nn.Conv2d(in_channels=filter_in, out_channels=filter_out,
                               kernel_size=3, stride=1, padding=1
        )

        if filter_in == filter_out:
            self.identity = lambda x : x 
        else: 
            self.identity = nn.Conv2d(in_channels=filter_in, out_channels=filter_out, 
                                    kernel_size=1, stride=1, padding=1
                                    )
        

    def forward(self, x):
        x = F.relu(self.bn1(x))
        x = self.conv1(x)
        x = F.relu(self.bn2(x))
        x = self.conv2(x)
        




# _Residual Layer 
class ResnetLayer(nn.Module):
    def __init__(self):
        super(ResnetLayer, self).__init__()

