import torch
from torchvision import transforms

import numpy as np
import pandas as pd
import os, sys, random
from tqdm import tqdm
from datetime import datetime

from modules.utils import load_yaml,save_yaml
from modules.datasets import MaskBaseDataset, TestDataset
from model.models import get_model

prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)

import warnings
warnings.filterwarnings('ignore')
if __name__ == '__main__':

    #Load Yaml
    config = load_yaml(os.path.join(prj_dir, 'config', 'test.yaml'))
    train_config = load_yaml(os.path.join(prj_dir, 'results', 'train', config['train_serial'], f'train.yaml'))

    pred_serial = config['train_serial'] + '_' + datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set random seed, deterministic
    torch.cuda.manual_seed(train_config['seed'])
    torch.manual_seed(train_config['seed'])
    np.random.seed(train_config['seed'])
    random.seed(train_config['seed'])
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
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(train_config['resize_size']),
    transforms.Normalize(mean=train_config['mean'],
                        std=train_config['std'])
    ])
    
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    test_dataset = TestDataset(img_paths, transform)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                batch_size=config['batch_size'],
                                num_workers=config['num_workers'],
                                shuffle=False,
                                drop_last=False)

    model_dir = os.path.join(prj_dir, 'results', 'train', config['train_serial'])
    check_point_path = os.path.join(model_dir, f'best_model.pt')

    # Adjust this line based on how you retrieve the model architecture and args from your config
    model = get_model(train_config[f'model']['architecture'])
    model = model(**train_config[f'model']['args'])
    model = model.to(device)

    # Load the checkpoint
    check_point = torch.load(check_point_path, map_location=torch.device("cpu"))
    model.load_state_dict(check_point['model'])
    model.eval()

 
    # Make predictions
    model_preds = []
    with torch.no_grad():
        for iter, img in enumerate(tqdm(test_dataloader)):
            img = img.to(device)
            
            batch_size = img.shape[0]
            # pred_value_1 = models[0](img)
            # pred_value_2 = models[1](img)
            # pred_value_3 = models[2](img)
            pred_value_1 = model(img, "mask")
            pred_value_2 = model(img, "gender")
            pred_value_3 = model(img, "age")            
            pred_value_1 = pred_value_1.argmax(dim=-1)
            pred_value_2 = pred_value_2.argmax(dim=-1)
            pred_value_3 = pred_value_3.argmax(dim=-1)
            pred_value = MaskBaseDataset.encode_multi_class(pred_value_1, pred_value_2, pred_value_3)
            model_preds.extend(pred_value.cpu().numpy())


    info["ans"] = model_preds
    save_path = os.path.join(pred_result_dir , "output.csv")
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")