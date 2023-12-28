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
from modules.datasets import CombinedDataset, MaskBaseDataset, MaskSplitByProfileDataset, ModifiedGenerationDataset, get_dataset_function
from modules.metrics import accuracy, age_class_f1Score, f1Score, gender_class_f1Score, get_metric_function, mask_class_f1Score
from modules.utils import load_yaml,save_yaml
from modules.logger import MetricAverageMeter,LossAverageMeter
from torchvision.transforms import v2

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

if __name__ == "__main__":
    
    train_serial = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_result_dir = os.path.join(prj_dir, 'results', 'train', train_serial)
    os.makedirs(train_result_dir, exist_ok=True)
    #data dir
    
    # Load config
    config_path = os.path.join(prj_dir, 'config', 'train.yaml')
    config = load_yaml(config_path)
    shutil.copy(config_path, os.path.join(train_result_dir,'train.yaml'))
    
    data_dir = config['train_dir']
    data_gen_dir = config['train_gen_dir']
    
    #wandb
    if config['wandb']:
        wandb.init(project=config["wandb_project"], config={
                    "learning_rate": config['optimizer']['args']['lr'],
                    "architecture": config['model']['architecture'],
                    "dataset": "MaskDaset",
                    "notes":config['wandb_note']
                    },
                   name = config['wandb_run'])
    else:
        wandb.init(mode="disabled")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_num'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device : ",device)
    
    
    # transform = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.Resize(config['resize_size']),
    # transforms.Normalize(mean=config['mean'],
    #                     std=config['std'])

                        
    # ])

    transform =  transforms.Compose([
                transforms.ToTensor(),
                v2.RandomHorizontalFlip(p=0.5),
                transforms.CenterCrop([360,256]),
                transforms.Resize(config['resize_size']),
                transforms.Normalize(mean=config['mean'],
                                    std=config['std'])
                ])


    if config['dataset'] == "baseDataset":
        dataset = get_dataset_function(config['dataset'])
        dataset = dataset(data_dir, transform,val_ratio=config['val_size'])
        
        train_dataset, val_dataset = dataset.split_dataset()
        
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], drop_last=config['drop_last'],num_workers=config['num_workers'])
        val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], drop_last=config['drop_last'],num_workers=config['num_workers'])
    else:
        dataset = get_dataset_function(config['dataset'])
        dataset = dataset(data_dir, transform,val_ratio=config['val_size'],seed=config['seed'])
        
        train_dataset, val_dataset = dataset.split_dataset()
        
        train_sampler = dataset.get_sampler('train')
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], drop_last=config['drop_last'],num_workers=config['num_workers'], sampler=train_sampler)
    
        valid_sampler = dataset.get_sampler('val')
        val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], drop_last=config['drop_last'],num_workers=config['num_workers'], sampler=valid_sampler)
        
    # dataset_tatin = MaskBaseDataset(data_dir, transform, val_ratio=config['val_size'])
    # dataset_generation = ModifiedGenerationDataset(data_gen_dir, transform, val_ratio=config['val_size'])

    num_classes = dataset.num_classes
    
    # combined_dataset = CombinedDataset(dataset_tatin, dataset_generation)

    # train_dataset, val_dataset = dataset.split_dataset()

    # train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'],drop_last=config['drop_last'],num_workers=config['num_workers'])
    # val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'],drop_last=config['drop_last'],num_workers=config['num_workers'])
    
    if config['model_custom']:
        model = get_model(config['model']['architecture'])
        model = model(**config['model']['args'])
    else:
        model = get_model(config['model']['architecture'])
        model = model(config['model']['architecture'], **config['model']['args'])
    model = model.to(device)
    wandb.watch(model)
    
    optimizer = get_optimizer(optimizer_str=config['optimizer']['name'])
    optimizer = optimizer(model.parameters(), **config['optimizer']['args'])
    
    scheduler = get_scheduler(scheduler_str=config['scheduler']['name'])
    scheduler = scheduler(optimizer=optimizer, **config['scheduler']['args'])
    
    loss_func = get_loss_function(loss_function_str=config['loss']['name'])
    # loss_func = loss_func(**config['loss']['args'])
    # loss_func = loss_func()
    
    metric_funcs = {metric_name:get_metric_function(metric_name) for metric_name in config['metrics']}
    max_f1_score = 0
  
    model.train()
    
    f1_score_lst = ["acc", "f1_score", "mask_f1_score", "gender_f1_score", "age_f1_score"]
    f1_class_score_lst = ["mask_class_f1_score", "gender_class_f1_score", "age_class_f1_score"]
    
    for epoch_id in tqdm(range(config['n_epochs'])):
        tic = time()
        train_loss = 0
        train_scores = {metric_name: 0 for metric_name, _ in metric_funcs.items() if metric_name in f1_score_lst}
        train_class_scores = {metric_name: np.array([0.,0.,0.]) for metric_name, _ in metric_funcs.items() if metric_name in f1_class_score_lst}
        train_class_cnt = {metric_name: np.array([0.,0.,0.]) for metric_name, _ in metric_funcs.items() if metric_name in f1_class_score_lst}
        
        for iter, (img, label) in enumerate(train_dataloader):
            img = img.to(device)
            mask_label, gender_label, age_label = MaskBaseDataset.decode_multi_class(label)
            mask_label, gender_label, age_label = mask_label.to(device), gender_label.to(device), age_label.to(device)

            batch_size = img.shape[0]

            pred_value_mask = model(img, "mask")
            pred_value_gender = model(img, "gender")
            pred_value_age = model(img, "age")

            loss_mask = loss_func(pred_value_mask, mask_label)
            loss_gender = loss_func(pred_value_gender, gender_label)
            loss_age = loss_func(pred_value_age, age_label)

            loss = 0.05 * loss_mask + 0.05 * loss_gender + 0.9 * loss_age

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Accuracy 계산
            mask_acc = 0
            gender_acc = 0
            age_acc = 0


            mask_acc += accuracy(pred_value_mask, mask_label) / len(train_dataloader)
            gender_acc += accuracy(pred_value_gender, gender_label) / len(train_dataloader)
            age_acc += accuracy(pred_value_age, age_label) / len(train_dataloader)
            train_scores['acc'] += (mask_acc + gender_acc + age_acc) / 3

            train_scores['mask_f1_score'] += f1Score(pred_value_mask, mask_label) / len(train_dataloader)
            train_scores['gender_f1_score'] += f1Score(pred_value_gender, gender_label) / len(train_dataloader)
            train_scores['age_f1_score'] += f1Score(pred_value_age, age_label) / len(train_dataloader)

            score_mask, cnt_mask = mask_class_f1Score(pred_value_mask, mask_label)
            train_class_scores["mask_class_f1_score"] += score_mask
            train_class_cnt["mask_class_f1_score"] += cnt_mask

            score_gender, cnt_gender = gender_class_f1Score(pred_value_gender, gender_label)
            train_class_scores["gender_class_f1_score"] += score_gender
            train_class_cnt["gender_class_f1_score"] += cnt_gender

            score_age, cnt_age = age_class_f1Score(pred_value_age, age_label)
            train_class_scores["age_class_f1_score"] += score_age
            train_class_cnt["age_class_f1_score"] += cnt_age


            train_loss += loss.item() / len(train_dataloader)
            
        for metric_name, _ in train_class_scores.items():
            for i in range(3):
                if train_class_scores[metric_name][i] != 0:
                    train_class_scores[metric_name][i] = train_class_scores[metric_name][i] / train_class_cnt[metric_name][i]    

        scheduler.step()

        train_scores['f1_score'] = (train_scores['mask_f1_score'] + train_scores['gender_f1_score'] + train_scores['age_f1_score']) / 3

            
        # Validation
        valid_loss = 0
        valid_scores = {metric_name: 0 for metric_name, _ in metric_funcs.items() if metric_name in f1_score_lst}
        valid_class_scores = {metric_name: np.array([0.,0.,0.]) for metric_name, _ in metric_funcs.items() if metric_name in f1_class_score_lst}
        valid_class_cnt = {metric_name: np.array([0.,0.,0.]) for metric_name, _ in metric_funcs.items() if metric_name in f1_class_score_lst}

        # if (iter % 20 == 0) or (iter == len(qd_train_dataloader)-1):
        model.eval()
        toc = time()
        train_time = toc- tic

        for img, label in val_dataloader:

            img = img.to(device)
            mask_label, gender_label, age_label = MaskBaseDataset.decode_multi_class(label)
            mask_label, gender_label, age_label = mask_label.to(device), gender_label.to(device), age_label.to(device)

            batch_size = img.shape[0]
            with torch.no_grad():

                pred_value_mask = model(img, "mask")
                pred_value_gender = model(img, "gender")
                pred_value_age = model(img, "age")

                loss_mask = loss_func(pred_value_mask, mask_label)
                loss_gender = loss_func(pred_value_gender, gender_label)
                loss_age = loss_func(pred_value_age, age_label)

                # Accuracy 계산
                mask_acc += accuracy(pred_value_mask, mask_label) / len(val_dataloader)
                gender_acc += accuracy(pred_value_gender, gender_label) / len(val_dataloader)
                age_acc += accuracy(pred_value_age, age_label) / len(val_dataloader)

                valid_scores['mask_f1_score'] += f1Score(pred_value_mask, mask_label) / len(val_dataloader)
                valid_scores['gender_f1_score'] += f1Score(pred_value_gender, gender_label) / len(val_dataloader)
                valid_scores['age_f1_score'] += f1Score(pred_value_age, age_label) / len(val_dataloader)

                score_mask, cnt_mask = mask_class_f1Score(pred_value_mask, mask_label)
                valid_class_scores["mask_class_f1_score"] += score_mask
                valid_class_cnt["mask_class_f1_score"] += cnt_mask

                score_gender, cnt_gender = gender_class_f1Score(pred_value_gender, gender_label)
                valid_class_scores["gender_class_f1_score"] += score_gender
                valid_class_cnt["gender_class_f1_score"] += cnt_gender

                score_age, cnt_age = age_class_f1Score(pred_value_age, age_label)
                valid_class_scores["age_class_f1_score"] += score_age
                valid_class_cnt["age_class_f1_score"] += cnt_age

                valid_loss_mask = loss_func(pred_value_mask, mask_label)
                valid_loss_gender = loss_func(pred_value_gender, gender_label)
                valid_loss_age = loss_func(pred_value_age, age_label)

                valid_loss += (valid_loss_mask + valid_loss_gender + valid_loss_age).item() / batch_size


            # valid_loss += loss.item() / batch_size
            
        for metric_name, _ in valid_class_scores.items():
            for i in range(3):
                if valid_class_scores[metric_name][i] != 0:
                    valid_class_scores[metric_name][i] = valid_class_scores[metric_name][i] / valid_class_cnt[metric_name][i]    

        valid_scores['acc'] = (mask_acc + gender_acc + age_acc) / 3
        valid_scores['f1_score'] = (valid_scores['mask_f1_score'] + valid_scores['gender_f1_score'] + valid_scores['age_f1_score']) / 3

        print("Epoch [%4d/%4d] | Train Loss %.4f | Train Acc %.4f | Train F1 %.4f | Valid Loss %.4f | Valid Acc %.4f | Valid F1 %.4f"  %
            (epoch_id, config['n_epochs'], train_loss, train_scores['acc'], train_scores['f1_score'], valid_loss, valid_scores['acc'], valid_scores['f1_score']))
        print("  train_mask_f1_score %.4f | label_0 %.4f | label_1 %.4f | label_2 %.4f" % (train_scores['mask_f1_score'], train_class_scores['mask_class_f1_score'][0], train_class_scores['mask_class_f1_score'][1], train_class_scores['mask_class_f1_score'][2]))
        print("  train_gender_f1_score %.4f | label_0 %.4f | label_1 %.4f" % (train_scores['gender_f1_score'], train_class_scores['gender_class_f1_score'][0], train_class_scores['gender_class_f1_score'][1]))
        print("  train_age_f1_score %.4f | label_0 %.4f | label_1 %.4f | label_2 %.4f" % (train_scores['age_f1_score'], train_class_scores['age_class_f1_score'][0], train_class_scores['age_class_f1_score'][1], train_class_scores['age_class_f1_score'][2]))
        print("  valid_mask_f1_score %.4f | label_0 %.4f | label_1 %.4f | label_2 %.4f" % (valid_scores['mask_f1_score'], valid_class_scores['mask_class_f1_score'][0], valid_class_scores['mask_class_f1_score'][1], valid_class_scores['mask_class_f1_score'][2]))
        print("  valid_gender_f1_score %.4f | label_0 %.4f | label_1 %.4f" % (valid_scores['gender_f1_score'], valid_class_scores['gender_class_f1_score'][0], valid_class_scores['gender_class_f1_score'][1]))
        print("  valid_age_f1_score %.4f | label_0 %.4f | label_1 %.4f | label_2 %.4f" % (valid_scores['age_f1_score'], valid_class_scores['age_class_f1_score'][0], valid_class_scores['age_class_f1_score'][1], valid_class_scores['age_class_f1_score'][2]))
        
        wandb.log({"train_time":train_time,"train_loss":train_loss,"train_acc":train_scores['acc'],"train_f1":train_scores['f1_score'], "valid_loss":valid_loss, "valid_acc":valid_scores['acc'], "valid_f1":valid_scores['f1_score'],
                   "train_mask_f1_score":train_scores['mask_f1_score'],"train_mask0_f1_score":train_class_scores['mask_class_f1_score'][0],"train_mask1_f1_score":train_class_scores['mask_class_f1_score'][1],"train_mask2_f1_score":train_class_scores['mask_class_f1_score'][2],
                   "train_age_f1_score":train_scores['age_f1_score'],"train_age0_f1_score":train_class_scores['age_class_f1_score'][0],"train_age1_f1_score":train_class_scores['age_class_f1_score'][1],"train_age2_f1_score":train_class_scores['age_class_f1_score'][2],
                   "train_gender_f1_score":train_scores['gender_f1_score'],"train_gender0_f1_score":train_class_scores['gender_class_f1_score'][0],"train_gender1_f1_score":train_class_scores['gender_class_f1_score'][1],
                   "valid_mask_f1_score":valid_scores['mask_f1_score'],"valid_mask0_f1_score":valid_class_scores['mask_class_f1_score'][0],"valid_mask1_f1_score":valid_class_scores['mask_class_f1_score'][1],"valid_mask2_f1_score":valid_class_scores['mask_class_f1_score'][2],
                   "valid_age_f1_score":valid_scores['age_f1_score'],"valid_age0_f1_score":valid_class_scores['age_class_f1_score'][0],"valid_age1_f1_score":valid_class_scores['age_class_f1_score'][1],"valid_age2_f1_score":valid_class_scores['age_class_f1_score'][2],
                   "valid_gender_f1_score":valid_scores['gender_f1_score'],"valid_gender0_f1_score":valid_class_scores['gender_class_f1_score'][0],"valid_gender1_f1_score":valid_class_scores['gender_class_f1_score'][1]})
        
        if max_f1_score < valid_scores['f1_score']:
            check_point = {
            'epoch': epoch_id + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None}
            
            torch.save(check_point,os.path.join(train_result_dir,f'model_{epoch_id}.pt'))
            torch.save(check_point,os.path.join(train_result_dir,f'best_model.pt'))
            early_stopping_count = 0
            max_f1_score = valid_scores['age_f1_score']
        else:
            early_stopping_count += 1
        
        if early_stopping_count >= config['early_stopping_count']:
            exit()

    # print(model)
    
    