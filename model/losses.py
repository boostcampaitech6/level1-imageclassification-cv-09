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



def cross_entropy_loss(output,target):
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
    def __init__(self, gamma=2, alpha=0.25, size_average=True, device='cuda:0',**kwargs):
        super(FocalLoss, self).__init__()
        """
        gamma(int) : focusing parameter.
        alpha(list) : alpha-balanced term.
        size_average(bool) : whether to apply reduction to the output.
        """
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.device = device

    def forward(self, inputs, targets):
        # input : N * C (btach_size, num_class)
        # target : N (batch_size)

        CE = F.cross_entropy(inputs, targets, reduction='none')  # -log(pt)
        pt = torch.exp(-CE)  # pt
        loss = (1 - pt) ** self.gamma * CE  # -(1-pt)^rlog(pt)

        if self.alpha is not None:
            alpha = torch.tensor(self.alpha, dtype=torch.float).to(self.device)
            # in case that a minority class is not selected when mini-batch sampling
            if len(self.alpha) != len(torch.unique(targets)):
                temp = torch.zeros(len(self.alpha)).to(self.device)
                temp[torch.unique(targets)] = alpha.index_select(0, torch.unique(targets))
                alpha_t = temp.gather(0, targets)
                loss = alpha_t * loss
            else:
                alpha_t = alpha.gather(0, targets)
                loss = alpha_t * loss

        if self.size_average:
            loss = torch.mean(loss)

        return loss
