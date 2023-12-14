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

from model.models import ResNet,ResNet1
from model.models import BasicBlock
from model.optimizers import get_optimizer
from model.losses import get_loss_function

from modules.datasets import MaskBaseDataset
from modules.metrics import get_metric_function
from modules.utils import load_yaml
from modules.logger import AverageMeter

prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)

seed = 111

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

import warnings
warnings.filterwarnings('ignore')

train_serial = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join(prj_dir, 'log', train_serial)
if __name__ == "__main__":
    os.makedirs(log_dir, exist_ok=True)
    #data dir
    
    # Load config
    config_path = os.path.join(prj_dir, 'config', 'train.yaml')
    config = load_yaml(config_path)
    shutil.copy(config_path, os.path.join(log_dir,'train.yaml'))
    
    data_dir = config['train_dir']
    
    #wandb
    if config['wandb']:
        wandb.init(project="boostcamp_Project1", config={
                    "learning_rate": 0.0003,
                    "architecture": "Resnet",
                    "dataset": "AN",
                    "notes":"No batchnorm in skipconnection"
                    },
                   name = "Focal Loss no maxpooling with no Yes batchnorm yes relu in skipconnection")
    else:
        wandb.init(mode="disabled")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_num'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device : ",device)
    
    
            
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 96)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])

    
    dataset = MaskBaseDataset(data_dir, transform)
    num_classes = MaskBaseDataset.num_classes
    
    train_dataset, val_dataset = dataset.split_dataset()
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    
    model = ResNet(3, 10).to(device)
    # model = ResNet1(BasicBlock, [3, 4, 6, 3]).to(device)
    wandb.watch(model)
    
    optimizer = get_optimizer(optimizer_str=config['optimizer']['name'])
    optimizer = optimizer(model.parameters(), **config['optimizer']['args'])
    
    scheduler = StepLR(optimizer, 20, gamma=0.5)
    loss_func = get_loss_function(loss_function_str=config['loss']['name'])
    # loss_func = loss_func(**config['loss']['args'])
    # loss_func = loss_func()
    
    acc_metric_func = get_metric_function('acc')
    f1_metric_func = get_metric_function('f1_score')
    max_f1_score = 0
    
    model.train()
    
    for epoch_id in range(config['n_epochs']):
        tic = time()
        train_loss, train_metric = AverageMeter(), AverageMeter()
        
        for iter, (img, label) in enumerate(tqdm(train_dataloader)):
            img = img.to(device)
            label = label.to(device)
            
            batch_size = img.shape[0]

            pred_value = model(img)

            loss = loss_func(pred_value, label)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Accuracy 계산
            acc = acc_metric_func(pred_value,label)
            f1 = f1_metric_func(pred_value,label)
            
            train_loss.update(loss.item(), batch_size)
            train_metric.update(acc,f1, batch_size)
            
        train_loss = train_loss.acc_avg
        train_acc = train_metric.acc_avg
        train_f1 = train_metric.f1_avg
        
        scheduler.step()
            
        # Validation
        valid_loss, valid_metric = AverageMeter(), AverageMeter()
        # if (iter % 20 == 0) or (iter == len(qd_train_dataloader)-1):
        model.eval()
        toc = time()
        train_time = toc- tic

        for img, label in val_dataloader:

            ##fill##
            img = img.to(device)
            label = label.to(device)
            batch_size = img.shape[0]
            with torch.no_grad():
                pred_value = model(img)
            loss = loss_func(pred_value, label)
            
            # Accuracy 계산
            acc = acc_metric_func(pred_value,label)
            f1 = f1_metric_func(pred_value,label)
            
                    
            valid_loss.update(loss.item(), batch_size)
            valid_metric.update(acc, f1, batch_size)
            
        valid_loss = valid_loss.acc_avg
        valid_acc = valid_metric.acc_avg
        valid_f1 = valid_metric.f1_avg
        
        # print("Epoch [%4d/%4d] | Train Loss %.4f | Train Acc %.4f | Valid Loss %.4f | Valid Acc %.4f" %
        #     (epoch_id, config['n_epochs'], train_loss, train_acc, valid_loss, valid_acc))
        print("Epoch [%4d/%4d] | Train Loss %.4f | Train Acc %.4f | Train F1 %.4f | Valid Loss %.4f | Valid Acc %.4f | Valid F1 %.4f"  %
            (epoch_id, config['n_epochs'], train_loss, train_acc, train_f1, valid_loss, valid_acc, valid_f1))
        wandb.log({"train_time":train_time,"train_loss":train_loss,"train_acc":train_acc,"train_f1":train_f1, "valid_loss":valid_loss, "valid_acc":valid_acc, "valid_f1":valid_f1})
        
        if max_f1_score < valid_f1:
            torch.save(model.state_dict(),os.path.join(log_dir,f'model_{epoch_id}.pth'))
            early_stopping_count = 0
            max_f1_score = valid_f1
        else:
            early_stopping_count += 1
        
        if early_stopping_count >= config['early_stopping_count']:
            exit()

    # print(model)
    
    