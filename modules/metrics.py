import torch
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

import numpy as np
import pandas as pd

def get_metric_function(metric_function_str):
    """
    Add metrics, weights for weighted score
    """

    if metric_function_str == 'acc':
        return accuracy
    elif metric_function_str == 'f1_score':
        return f1Score
    elif metric_function_str == 'mask_f1_score':
        return mask_f1Score
    elif metric_function_str == 'gender_f1_score':
        return gender_f1Score
    elif metric_function_str == 'age_f1_score':
        return age_f1Score
    elif metric_function_str == 'mask_class_f1_score':
        return mask_class_f1Score
    elif metric_function_str == 'gender_class_f1_score':
        return gender_class_f1Score
    elif metric_function_str == 'age_class_f1_score':
        return age_class_f1Score
    elif metric_function_str == 'mask_age_f1_score':
        return mask_age_f1_score
    
def accuracy(output,target):
    pred_label = torch.argmax(output, 1)
    # print(pred_label,target)
    acc = (pred_label == target).sum().item() / target.shape[0]
    return acc

def f1Score(output,target):
    pred_label = torch.argmax(output, 1).cpu()
    target = target.cpu()
    return f1_score(pred_label,target, average='macro')


# def mask_f1Score(output,target):
#     labels = ["class 0", "class 1", "class 2"]
#     pred_label = torch.argmax(output, 1).cpu()
#     pred_label = (pred_label // 6) % 3
#     target = (target // 6) % 3
#     target = target.cpu()
    
#     data = classification_report(target, pred_label, target_names=labels)
    
    
#     return data


def mask_f1Score(output,target):
    pred_label = torch.argmax(output, 1).cpu()
    pred_label = (pred_label // 6) % 3
    target = (target // 6) % 3
    target = target.cpu()
    return f1_score(pred_label,target, average='macro')


def gender_f1Score(output,target):
    pred_label = torch.argmax(output, 1).cpu()
    pred_label = (pred_label // 3) % 2
    target = (target // 3) % 2
    target = target.cpu()
    return f1_score(pred_label,target, average='macro')


def age_f1Score(output,target):
    pred_label = torch.argmax(output, 1).cpu()
    pred_label = pred_label % 3
    target = target % 3
    target = target.cpu()
    return f1_score(pred_label,target, average='macro')   


def mask_class_f1Score(output,target):
    pred_label = torch.argmax(output, 1).cpu()
    pred_label = (pred_label // 6) % 3
    target = (target // 6) % 3
    target = target.cpu()
    
    arr = [0.,0.,0.]
    cnt = [0,0,0]
    all_unique = torch.cat([pred_label,target]).unique()
    for i in range(3):
        if i in all_unique:
            arr[i] = f1_score(pred_label,target, average="micro", labels=[i])
            cnt[i] += 1
        else:
            pass
    return np.array(arr), np.array(cnt)


def gender_class_f1Score(output,target):
    pred_label = torch.argmax(output, 1).cpu()
    pred_label = (pred_label // 3) % 2
    target = (target // 3) % 2
    target = target.cpu()
    arr = [0.,0.,0.]
    cnt = [0,0,0]
    all_unique = torch.cat([pred_label,target]).unique()
    for i in range(3):
        if i in all_unique:
            arr[i] = f1_score(pred_label,target, average="micro", labels=[i])
            cnt[i] += 1
        else:
            pass
    return np.array(arr), np.array(cnt)
    

def age_class_f1Score(output,target):
    pred_label = torch.argmax(output, 1).cpu()
    pred_label = pred_label % 3
    target = target % 3
    target = target.cpu()
    arr = [0.,0.,0.]
    cnt = [0,0,0]
    all_unique = torch.cat([pred_label,target]).unique()
    for i in range(3):
        if i in all_unique:
            arr[i] = f1_score(pred_label,target, average="micro", labels=[i])
            cnt[i] += 1
        else:
            pass
    return np.array(arr), np.array(cnt)

def mask_age_f1_score(output,target):
    pred_label = torch.argmax(output, 1).cpu()
    pred_mask_label = (pred_label // 6) % 3
    pred_age_label = pred_label % 3
    
    target = target.cpu()
    target_mask_label = (target // 6) % 3
    target_age_label = target % 3
    
    pred_mask_age_label = pred_mask_label * 3 + pred_age_label
    target_mask_age_label = target_mask_label * 3 + target_age_label
    
    all_unique = torch.cat([pred_mask_age_label,target_mask_age_label]).unique()
    arr = [0.,0.,0., 0.,0.,0., 0.,0.,0.]
    cnt = [0,0,0, 0,0,0, 0,0,0]
    for i in range(9):
        if i in all_unique:
            arr[i] = f1_score(pred_mask_age_label,target_mask_age_label, average="micro", labels=[i])
            cnt[i] += 1
        else:
            pass
    return np.array(arr), np.array(cnt)