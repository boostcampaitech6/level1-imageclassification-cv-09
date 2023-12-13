import torch
from sklearn.metrics import f1_score

def get_metric_function(metric_function_str):
    """
    Add metrics, weights for weighted score
    """

    if metric_function_str == 'acc':
        return accuracy
    elif metric_function_str == 'f1_score':
        return f1Score
    
def accuracy(output,target):
    
    pred_label = torch.argmax(output, 1)
    # print(pred_label,target)
    acc = (pred_label == target).sum().item() / target.shape[0]
    return acc

def f1Score(output,target):
    pred_label = torch.argmax(output, 1).cpu()
    target = target.cpu()
    return f1_score(pred_label,target, average='macro')