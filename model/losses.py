import torch
import torch.nn as nn
from torch.nn import functional as F


def get_loss_function(loss_function_str: str):

    if loss_function_str == 'MeanCCELoss':

        return CCE

    elif loss_function_str == 'GDLoss':

        return GeneralizedDiceLoss
    elif loss_function_str == "FocalLoss":

        return FocalLoss

    elif loss_function_str == "Cross_entropy":
        return cross_entropy_loss
    
    elif loss_function_str == "FocalTverskyLoss":
        return FocalTverskyLoss
    
    elif loss_function_str == "TverskyLoss":
        return TverskyLoss

class cross_entropy_loss(nn.Module):
    def __init__(self, weight, **kwargs):
        super(cross_entropy_loss, self).__init__()
        
    def forward(self, output,target):
        cross_entropy = nn.CrossEntropyLoss()
        return cross_entropy(output,target)


class CCE(nn.Module):

    def __init__(self, weight, **kwargs):
        super(CCE, self).__init__()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight = torch.Tensor(weight).to(device)

    def forward(self, inputs, targets):
        
        loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        unique_values, unique_counts = torch.unique(targets, return_counts=True)
        selected_weight = torch.index_select(input=self.weight, dim=0, index=unique_values)

        numerator = loss.sum()                               # weighted losses
        denominator = (unique_counts*selected_weight).sum()  # weigthed counts

        loss = numerator/denominator

        return loss


class GeneralizedDiceLoss(nn.Module):
    
    def __init__(self, **kwargs):
        super(GeneralizedDiceLoss, self).__init__()
        self.scaler = nn.Softmax(dim=1)  # Softmax for loss

    def forward(self, inputs, targets):

        targets = targets.contiguous()
        targets = torch.nn.functional.one_hot(targets.to(torch.int64), inputs.size()[1])  # B, H, W, C

        inputs = inputs.contiguous()
        inputs = self.scaler(inputs)
        inputs = inputs.permute(0, 2, 3, 1)  # B, H, W, C

        w = 1. / (torch.sum(targets, (0, 1, 2)) ** 2 + 1e-9)

        numerator = targets * inputs
        numerator = w * torch.sum(numerator, (0, 1, 2))
        numerator = torch.sum(numerator)

        denominator = targets + inputs
        denominator = w * torch.sum(denominator, (0, 1, 2))
        denominator = torch.sum(denominator)

        dice = 2. * (numerator + 1e-9) / (denominator + 1e-9)

        return 1. - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
    
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        
class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.75):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        # Assuming y_pred is softmax output, so we don't need to apply softmax
        # If not, apply softmax or log_softmax depending on your requirements
        y_pred = F.softmax(y_pred,dim=1)
        
        # Convert y_true to one-hot format
        y_true_one_hot = F.one_hot(y_true, num_classes=y_pred.size(1)).float()

        # True Positives, False Positives & False Negatives
        tp = torch.sum(y_pred * y_true_one_hot, dim=0)
        fp = torch.sum(y_pred * (1 - y_true_one_hot), dim=0)
        fn = torch.sum((1 - y_pred) * y_true_one_hot, dim=0)

        # Tversky index for each class
        tversky = tp / (tp + self.alpha * fp + self.beta * fn)

        # Focal Tversky Loss
        focal_tversky_loss = torch.sum(torch.pow((1 - tversky), 1 / self.gamma))

        return focal_tversky_loss
    
class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        # Convert y_true to one-hot format if it's not already
        y_pred = F.softmax(y_pred,dim=1)
        y_true_one_hot = F.one_hot(y_true, num_classes=y_pred.size(1)).float()

        # Flatten the tensors to shape (batch_size*num_classes, )
        y_true_flat = y_true_one_hot.view(-1)
        y_pred_flat = y_pred.view(-1)

        # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
        tp = torch.sum(y_pred_flat * y_true_flat)
        fp = torch.sum(y_pred_flat * (1 - y_true_flat))
        fn = torch.sum((1 - y_pred_flat) * y_true_flat)

        # Tversky index
        tversky_index = tp / (tp + self.alpha * fp + self.beta * fn)

        # Tversky loss
        tversky_loss = 1 - tversky_index

        return tversky_loss