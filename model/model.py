import torch.nn as nn
import torch
import torchvision

class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # get resnet model
        self.resnet = torchvision.models.resnet18(weights=None)
        #changed the first conv layer to make it accept gray scale images
        self.resnet.conv1 = nn.Conv2d(1,64,3)
        self.fc_in_features = self.resnet.fc.in_features
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        # pass the concatenation to the linear layers
        output1 = self.fc(output1)
        output2 = self.fc(output2)
        
        return output1,output2
    
    def forward_128(self, input):
        output= self.forward_once(input)
        output = self.fc(output)
        return output