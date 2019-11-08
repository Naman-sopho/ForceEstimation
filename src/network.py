import torch
import torch.nn as nn


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        ## 
        # Linear layer with tansig activation.
        #
        # Input Size 12, one set of Postion, Velocity
        # measurements from each of the 6 joints.
        #
        # Output Size 100
        ##
        self.layer1 = nn.Linear(12, 100, bias=True)

        ##
        # Linear Layer
        # 
        # Output Size 1, Torque estimate for this joint.
        ##
        self.layer2 = nn.Linear(100, 1, bias=True)

    def forward(self, x):
        layer1 = self.layer1(x)
        layer1 = torch.tanh(layer1)

        layer2 = self.layer2(layer1)

        output = layer2

        return output

