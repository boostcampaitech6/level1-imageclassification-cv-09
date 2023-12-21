import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

from torchvision import transforms, utils

import os, sys, shutil,itertools,random

import matplotlib.pyplot as plt

from datetime import datetime
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm

import wandb

from model.models import ResNet
from model.models import BasicBlock
from model.optimizers import get_optimizer
from model.losses import get_loss_function
from model.models import get_model

from modules.schedulers import get_scheduler
from modules.datasets import CombinedDataset, MaskBaseDataset, MaskSplitByProfileDataset, ModifiedGenerationDataset
from modules.metrics import get_metric_function
from modules.datasets import get_dataset_function
from modules.transforms import get_transform_function
from modules.utils import load_yaml,save_yaml
from modules.logger import MetricAverageMeter,LossAverageMeter

prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    
    config = load_yaml(os.path.join(prj_dir, 'config', 'ensemble.yaml'))
    
    train_id_list = config['train_serial'].split()
    train_config = [None for _ in range(len(train_id_list))]
    
    for idx, train_id in enumerate(train_id_list):
        train_config[idx] = load_yaml(os.path.join(prj_dir, 'results', 'train', train_id, 'train.yaml'))
    
    pred_serial = config['train_serial'] + '_' + datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set random seed, deterministic
    torch.cuda.manual_seed(train_config[0]['seed'])
    torch.manual_seed(train_config[0]['seed'])
    np.random.seed(train_config[0]['seed'])
    random.seed(train_config[0]['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    #Device set
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_num'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #result_dir
    pred_result_dir = os.path.join(prj_dir, 'results', 'pred', pred_serial)
    os.makedirs(pred_result_dir, exist_ok=True)
    
    data_dir = config['test_dir']
    img_root = os.path.join(data_dir, "images")
    info_path = os.path.join(data_dir, "info.csv")
    info = pd.read_csv(info_path)
    
    
    model = [None for _ in range(len(train_id_list))]
    transform = get_transform_function(train_config[idx]['transform'],train_config[idx])
        
    dataset = get_dataset_function(train_config[0]['dataset'])
    dataset = dataset(data_dir, transform,val_ratio=train_config[0]['val_size'],seed=train_config[0]['seed'], drop_age_mode=train_config[0]["drop_age_mode"], drop_age=train_config[0]["drop_age"])
        
    train_dataset, val_dataset = dataset.split_dataset()
        
    train_sampler = dataset.get_sampler('train')
    valid_sampler = dataset.get_sampler('val')
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], drop_last=config['drop_last'],num_workers=config['num_workers'], sampler=valid_sampler)
        
    for idx in range(len(train_id_list)):
        if train_config[idx]['model_custom']:
            model[idx] = get_model(train_config[idx]['model']['architecture'])
            model[idx] = model(**train_config[idx]['model']['args'])
        else:
            model[idx] = get_model(train_config[idx]['model']['architecture'])
            model[idx] = model[idx](train_config[idx]['model']['architecture'], **train_config[idx]['model']['args'])
        model[idx] = model[idx].to(device)
    
        print(f"Load model architecture: {train_config[idx]['model']['architecture']}")
    
        check_point_path = os.path.join(prj_dir, 'results', 'train', train_id_list[idx], 'best_model.pt')
        check_point = torch.load(check_point_path,map_location=torch.device("cpu"))
        model[idx].load_state_dict(check_point['model'])
        model[idx].eval()
    
        # Save config
        save_yaml(os.path.join(pred_result_dir, f'{train_id_list[idx]}_train.yaml'), train_config[idx])
        
    save_yaml(os.path.join(pred_result_dir, 'predict.yaml'), config)
    
    metric_funcs = {metric_name:get_metric_function(metric_name) for metric_name in config['metrics']}
    max_f1_score = 0
  
    model.train()
    
    f1_score_lst = ["acc", "f1_score", "mask_f1_score", "gender_f1_score", "age_f1_score"]
    f1_class_score_lst = ["mask_class_f1_score", "gender_class_f1_score", "age_class_f1_score"]
    f1_mask_age_lst= ["mask_age_f1_score"]
    

    tic = time()
    train_loss = 0
    train_scores = {metric_name: 0 for metric_name, _ in metric_funcs.items() if metric_name in f1_score_lst}
    train_class_scores = {metric_name: np.array([0.,0.,0.]) for metric_name, _ in metric_funcs.items() if metric_name in f1_class_score_lst}
    train_class_cnt = {metric_name: np.array([0.,0.,0.]) for metric_name, _ in metric_funcs.items() if metric_name in f1_class_score_lst}
    train_mask_class_score = {metric_name: np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.]) for metric_name, _ in metric_funcs.items() if metric_name in f1_mask_age_lst}
    train_mask_class_cnt = {metric_name: np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.]) for metric_name, _ in metric_funcs.items() if metric_name in f1_mask_age_lst}
    
        
    # Validation
    valid_loss = 0
    valid_scores = {metric_name: 0 for metric_name, _ in metric_funcs.items() if metric_name in f1_score_lst}
    valid_class_scores = {metric_name: np.array([0.,0.,0.]) for metric_name, _ in metric_funcs.items() if metric_name in f1_class_score_lst}
    valid_class_cnt = {metric_name: np.array([0.,0.,0.]) for metric_name, _ in metric_funcs.items() if metric_name in f1_class_score_lst}
    valid_mask_class_score = {metric_name: np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.]) for metric_name, _ in metric_funcs.items() if metric_name in f1_mask_age_lst}
    valid_mask_class_cnt = {metric_name: np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.]) for metric_name, _ in metric_funcs.items() if metric_name in f1_mask_age_lst}

    # if (iter % 20 == 0) or (iter == len(qd_train_dataloader)-1):
    model.eval()
    toc = time()
    train_time = toc- tic
    
    for img, label in val_dataloader:

        ##fill##
        img = img.to(device)
        label = label.to(device)
        batch_size = img.shape[0]
        pred_value = []
        with torch.no_grad():
            for idx in range(len(train_id_list)):
                pred_value.append(model[idx](img))
                
        if config['ensemble'] == "soft":
            pass
        elif config['ensemble'] == "hard":
            pass
        
        # Accuracy 계산
        for metric_name, metric_func in metric_funcs.items():
            if metric_name in f1_score_lst:
                valid_scores[metric_name] += metric_func(pred_value, label) / len(val_dataloader)
            elif metric_name in f1_class_score_lst:
                score, cnt = metric_func(pred_value, label)
                valid_class_scores[metric_name] += score
                valid_class_cnt[metric_name] += cnt
            elif metric_name in f1_mask_age_lst:
                score, cnt = metric_func(pred_value, label)
                valid_mask_class_score[metric_name] += score
                valid_mask_class_cnt[metric_name] += cnt
        
    for metric_name, _ in valid_class_scores.items():
        for i in range(3):
            if valid_class_scores[metric_name][i] != 0:
                valid_class_scores[metric_name][i] = valid_class_scores[metric_name][i] / valid_class_cnt[metric_name][i]
                
    for metric_name, _ in valid_mask_class_score.items():
        for i in range(9):
            if valid_mask_class_score[metric_name][i] != 0:
                valid_mask_class_score[metric_name][i] = valid_mask_class_score[metric_name][i] / valid_mask_class_cnt[metric_name][i]
        
    # print("Epoch [%4d/%4d] | Train Loss %.4f | Train Acc %.4f | Valid Loss %.4f | Valid Acc %.4f" %
    #     (epoch_id, config['n_epochs'], train_loss, train_acc, valid_loss, valid_acc))
    print("Valid Acc %.4f | Valid F1 %.4f"  % (valid_scores['acc'], valid_scores['f1_score']))
    print("  valid_mask_f1_score %.4f | label_0 %.4f | label_1 %.4f | label_2 %.4f" % (valid_scores['mask_f1_score'], valid_class_scores['mask_class_f1_score'][0], valid_class_scores['mask_class_f1_score'][1], valid_class_scores['mask_class_f1_score'][2]))
    print("  valid_gender_f1_score %.4f | label_0 %.4f | label_1 %.4f" % (valid_scores['gender_f1_score'], valid_class_scores['gender_class_f1_score'][0], valid_class_scores['gender_class_f1_score'][1]))
    print("  valid_age_f1_score %.4f | label_0 %.4f | label_1 %.4f | label_2 %.4f" % (valid_scores['age_f1_score'], valid_class_scores['age_class_f1_score'][0], valid_class_scores['age_class_f1_score'][1], valid_class_scores['age_class_f1_score'][2]))
    print("  valid_mask0_age0_f1_score %.4f | valid_mask0_age1_f1_score %.4f | valid_mask0_age2_f1_score %.4f | valid_mask1_age0_f1_score %.4f | valid_mask1_age1_f1_score %.4f | valid_mask1_age2_f1_score %.4f | valid_mask2_age0_f1_score %.4f | valid_mask2_age1_f1_score %.4f | valid_mask2_age2_f1_score %.4f" % tuple(x for x in valid_mask_class_score['mask_age_f1_score']))
    
    new_wandb_metric_dict = {"valid_mask_f1_score":valid_scores['mask_f1_score'],"valid_mask0_f1_score":valid_class_scores['mask_class_f1_score'][0],"valid_mask1_f1_score":valid_class_scores['mask_class_f1_score'][1],"valid_mask2_f1_score":valid_class_scores['mask_class_f1_score'][2],
                "valid_age_f1_score":valid_scores['age_f1_score'],"valid_age0_f1_score":valid_class_scores['age_class_f1_score'][0],"valid_age1_f1_score":valid_class_scores['age_class_f1_score'][1],"valid_age2_f1_score":valid_class_scores['age_class_f1_score'][2],
                "valid_gender_f1_score":valid_scores['gender_f1_score'],"valid_gender0_f1_score":valid_class_scores['gender_class_f1_score'][0],"valid_gender1_f1_score":valid_class_scores['gender_class_f1_score'][1]}

    for i in range(9):
        new_wandb_metric_dict[f"valid_mask{i//3}_age{i%3}_f1_score"] = valid_mask_class_score['mask_age_f1_score'][i]
                
        wandb.log(new_wandb_metric_dict)
        

    