import torch


def get_metric_function(metric_function_str):
    """
    Add metrics, weights for weighted score
    """

    if metric_function_str == 'acc':
        return accuracy
    elif metric_function_str == 'f1_score':
        return 
    
def accuracy(output,target):
    
    pred_label = torch.argmax(output, 1)
    # print(pred_label,target)
    acc = (pred_label == target).sum().item() / target.shape[0]
    return acc