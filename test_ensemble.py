import torch
from torchvision import transforms

import numpy as np
import pandas as pd
import os, sys, random
from tqdm import tqdm
from datetime import datetime

from modules.transforms import get_transform_function
from modules.utils import load_yaml,save_yaml
from modules.datasets import TestDataset
from model.models import get_model

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
    
    
    model = [None for _ in range(len(train_id_list))]
    data_dir = config['test_dir']
    img_root = os.path.join(data_dir, "images")
    info_path = os.path.join(data_dir, "info.csv")
    info = pd.read_csv(info_path)
    
    transform = get_transform_function(train_config[0]['transform'],train_config[0])
    
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    test_dataset = TestDataset(img_paths, transform)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                batch_size=config['batch_size'],
                                num_workers=config['num_workers'],
                                shuffle=False,
                                drop_last=False)
        
    for idx in range(len(train_id_list)):
        if train_config[idx]['model_custom']:
            model[idx] = get_model(train_config[idx]['model']['architecture'])
            model[idx] = model[idx](**train_config[idx]['model']['args'])
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
    
    preds = []
    with torch.no_grad():
        for iter, img in enumerate(tqdm(test_dataloader)):
            img = img.to(device)
            
            batch_size = img.shape[0]
            pred_value = []
            temp_value = 0
                
            if config['ensemble'] == "hard":
                with torch.no_grad():
                    for idx in range(len(train_id_list)):
                        pred_value.append(torch.nn.functional.one_hot(model[idx](img).argmax(dim=-1),18))
                temp_value = pred_value[0]
                for idx in range(len(train_id_list))[1:]:
                    temp_value = temp_value + pred_value[idx]
                pred_value = temp_value.argmax(dim=-1)
            elif config['ensemble'] == "soft":
                with torch.no_grad():
                    for idx in range(len(train_id_list)):
                        pred_value.append(model[idx](img))
                temp_value = pred_value[0]
                for idx in range(len(train_id_list))[1:]:
                    temp_value = temp_value + pred_value[idx]
                pred_value = temp_value.argmax(dim=-1)
            preds.extend(pred_value.cpu().numpy())
    info["ans"] = preds
    save_path = os.path.join(pred_result_dir , "output.csv")
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")
        

    