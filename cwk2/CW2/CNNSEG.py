import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNSEG(nn.Module): # Define your model
    def __init__(self):
        super(CNNSEG, self).__init__()
        self.cnn_layers = nn.Sequential()
        # fill in the constructor for your model here
    def forward(self, x):
        # fill in the forward function for your model here
        return x

model = CNNSEG() # We can now create a model using your defined segmentation model