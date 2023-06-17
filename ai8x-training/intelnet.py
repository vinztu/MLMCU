"""
IntelNet network description
"""
from signal import pause
from torch import nn, cat

import ai8x

import matplotlib
import matplotlib.pyplot as plt

"""
Network description class
"""
class intelnet_model(nn.Module):
    def __init__(self, num_classes=6, dimensions=(64, 64), num_channels=3, bias=True, **kwargs):
        super().__init__()

        # assert dimensions[0] == dimensions[1]  # Only square supported

        # Keep track of image dimensions so one constructor works for all image sizes
        dim_x, dim_y = dimensions

        self.conv1 = ai8x.FusedConv2dReLU(in_channels = num_channels, out_channels = 32, 
                                          kernel_size = 3, stride = 1,
                                          padding=0, bias=True, batchnorm = 'NoAffine', **kwargs)

        dim_x -= 2
        dim_y -= 2
        
      
    
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(in_channels = 32, out_channels = 64, 
                                          kernel_size = 3, pool_stride = 2,
                                          padding=0, bias=True, batchnorm = "NoAffine", **kwargs)
        
        dim_x //= 2  # pooling, padding 0
        dim_y //= 2
        
        dim_x -= 2  # pooling, padding 0
        dim_y -= 2
        # conv padding 1 -> no change in dimensions

        

        self.conv3 = ai8x.FusedConv2dReLU(in_channels = 64, out_channels = 16, stride = 1, kernel_size = 3,
                                          padding=1, bias=True, batchnorm = None,  **kwargs)
        
        
        self.conv5 = ai8x.FusedConv2dReLU(in_channels = 64, out_channels = 16, stride = 1, kernel_size = 1,
                                          padding=0, bias=True, batchnorm = None,  **kwargs)
        

        
        self.conv6 = ai8x.FusedMaxPoolConv2dReLU(in_channels = 32, out_channels = 16, kernel_size = 3, pool_stride = 2,
                                          padding=0, bias=True, batchnorm = 'NoAffine',  **kwargs)
        
        dim_x //= 2  # pooling, padding 0
        dim_y //= 2
        
        dim_x -= 2  # pooling, padding 0
        dim_y -= 2
        
        
        self.conv7 = ai8x.FusedConv2dReLU(in_channels = 16, out_channels = 8, kernel_size = 3,
                                          padding=0, bias=True, batchnorm = None,  **kwargs)
        
        dim_x -= 2  # pooling, padding 0
        dim_y -= 2   
        
        

        self.fc1 = ai8x.FusedLinearReLU(dim_x*dim_y*8, 32, bias=True, **kwargs)
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = ai8x.Linear(32, num_classes, wide=True, bias=True, **kwargs)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    """
    Assemble the model
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        
        x = self.conv1(x)
        
        x = self.conv2(x)
        
        ####
        x1 = self.conv3(x)
        
        x2 = self.conv5(x)
        #########
        
        x = cat((x1, x2),1)
        
        x = self.conv6(x)
        x = self.conv7(x)

        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        # Loss chosed, CrossEntropyLoss, takes softmax into account already func.log_softmax(x, dim=1))

        return x


def intelnet(pretrained=False, **kwargs):
    """
    Constructs a IntelNet model.
    """
    assert not pretrained
    return intelnet_model(**kwargs)

"""
Network description
"""
models = [
    {
        'name': 'intelnet',
        'min_input': 1,
        'dim': 2,
    }
]

#if __name__ == '__main__':
#    
#    model = intelnet()
#    
#    x = model(torch.randn(1,3,64,64))