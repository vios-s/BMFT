import torch
import torch.nn as nn
import torchvision.models as models


sigmoid = torch.nn.Sigmoid()

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * 0.1


class ClassificationHead(nn.Module):
    # Define model element
    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize a `ClassificationHead` module.

        :param in_features: The number of input features.
        :param out_features: The number of output features.
        """
        super(ClassificationHead, self).__init__()

        self.layer = nn.Linear(in_features, out_features)
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.layer(self.dropout(x)))
        
        return x
    
    
class AuxiliaryHead(nn.Module):
    # Define model element
    def __init__(self, in_features: int, num_aux: int) -> None:
        """Initialize a `ClassificationHead` module.

        :param in_features: The number of input features.
        :param num_classes: The number of output features.
        """
        super(AuxiliaryHead, self).__init__()

        self.layer = nn.Linear(in_features, num_aux)
        self.activation = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_aux = self.layer(x).squeeze()
        px_aux = self.activation(x_aux)
        
        return x_aux, px_aux
    
    
class AuxiliaryHead2(nn.Module):
    def __init__(self, in_features: int, num_aux: int) -> None:
        """Initialize a `ClassificationHead` module.

        :param in_features: The number of input features.
        :param num_classes: The number of output features.
        """
        super(AuxiliaryHead2, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_aux)
        )
        self.activation = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_aux = self.layer(x).squeeze()
        px_aux = self.activation(x_aux)
        
        return x_aux, px_aux
    

class Extractor(nn.Module):
    def __init__(self, weights=True):
        """
        ResNet-50 feature extractor
        """
        super(Extractor, self).__init__()

        self.enet = models.resnet50(weights=weights)
        self.dropout = nn.Dropout(p=0.5)
        in_ch = self.enet.fc.in_features
        self.enet.fc = nn.Identity()
        
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enet(x)
        
        return x
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Assigning feature representation to new variable to allow it
        to be pulled out and passed into auxiliary head

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        feat_out = self.extract(x).squeeze(-1).squeeze(-1)
        
        return feat_out


class Inception(Extractor):
    def __init__(self, weights=True):
        """
        Inception-V3 feature extractor
        """
        super(Inception, self).__init__()
        self.enet = models.inception_v3(weights=weights)
        self.enet.aux_logits = False
        self.dropouts = nn.Dropout(0.5)
        in_ch = self.enet.fc.in_features
        self.enet.fc = nn.Identity()
    

class DenseNet(Extractor):
    def __init__(self, weights=True):
        """
        DenseNet-161 feature extractor
        """
        super(DenseNet, self).__init__()

        self.enet = models.densenet161(weights=weights)
        self.dropout = nn.Dropout(p=0.5)
        in_ch = self.enet.classifier.in_features
        self.enet.classifier = nn.Identity()
        

class ResNext101(Extractor):
    def __init__(self, weights=True):
        """
        ResNext-101 feature extractor
        """
        super(ResNext101, self).__init__()

        self.enet = models.resnext101_32x8d(weights=weights)
        self.dropout = nn.Dropout(p=0.5)
        in_ch = self.enet.fc.in_features
        self.enet.fc = nn.Identity()
        

class ResNet101(Extractor):
    def __init__(self, weights=True):
        """
        ResNet-101 feature extractor
        """
        super(ResNet101, self).__init__()

        self.enet = models.resnet101(weights=weights)
        self.dropout = nn.Dropout(p=0.5)
        in_ch = self.enet.fc.in_features
        self.enet.fc = nn.Identity()
        
        
class EfficientNet(Extractor):
    def __init__(self, weights=True):
        """
        EfficientNet-B7 feature extractor
        """
        super(EfficientNet, self).__init__()

        # self.enet = geffnet.create_model('efficientnet_b3', pretrained=True)
        self.enet = models.efficientnet_b3(weights=weights)
        self.dropout = nn.Dropout(p=0.5)
        in_ch = self.enet.classifier[1].in_features
        self.enet.classifier[1] = nn.Identity()


class ResNet18(Extractor):
    def __int__(self, weights=True):
        """
        ResNet18 feature extractor
        """
        super(EfficientNet, self).__init__()

        # self.enet = geffnet.create_model('efficientnet_b3', pretrained=True)
        self.enet = models.resnet18(weights=weights)
        self.dropout = nn.Dropout(p=0.5)
        in_ch = self.enet.fc.in_features
        self.enet.fc = nn.Identity()
