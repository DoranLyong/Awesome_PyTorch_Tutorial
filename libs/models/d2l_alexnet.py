"""
Code source : https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch09_Modern_Convolutional_Networks/AlexNet.ipynb
"""

import torch 
import torch.nn as nn 


class d2l_AlexNet(nn.Module):
    def __init__(self,  num_classes=10,loss='softmax' ):
        super(d2l_AlexNet,self).__init__() 

        self.loss = loss

        self.model = nn.Sequential(
                    # 11x11 Conv, s=4, #96 
                    nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=1 ),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),

                    # 5x5 Conv, #256 
                    nn.Conv2d(96, 256, kernel_size=5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2) ,

                    # 3x3 Conv, #384 
                    nn.Conv2d(256, 384, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),

                    # 3x3 Conv, #384 
                    nn.Conv2d(384, 384, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),

                    # 3x3 Conv, #384 
                    nn.Conv2d(384, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),

                    # Linear 
                    nn.Flatten(),
                    nn.Dropout(p=0.5, inplace=True),
                    nn.Linear(in_features=6400, out_features=4096),
                    nn.ReLU() ,

                    # Linear 
                    nn.Dropout2d(p=0.5, inplace=True),
                    nn.Linear(in_features=4096, out_features=4096),
                    nn.ReLU(),

                    # Classifier 
                    nn.Linear(in_features=4096, out_features=num_classes),
                    )

    def forward(self, x):
        return self.model(x)


