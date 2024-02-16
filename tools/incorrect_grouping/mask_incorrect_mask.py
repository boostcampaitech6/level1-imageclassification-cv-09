import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm.notebook import tqdm
from time import time

import matplotlib.pyplot as plt
#import seaborn as sns
import multiprocessing as mp

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

data_dir = 'incorrect_image_gen\step1_resize'
file_list = os.listdir(data_dir)
file_list.sort()
cc = len(file_list)

maskclass = ['mask_sample']
choosemask = maskclass[0]

foreground = Image.open(f"incorrect_image_gen\step0_amask_images\{choosemask}.png")
save_dir = 'incorrect_image_gen\step2_incorrect_masked'
for i in range(0,cc,8):
    
    img_id = file_list[i]
    img_path = os.path.join(data_dir, img_id)
    
    img = Image.open(img_path)
    img_arr = np.array(img)
    # 얼굴 찾기
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces = face_cascade.detectMultiScale(img_arr)
    for (x, y, w, h) in faces:
        crop_box = (x, y, x + w, y + h)

    background = img.copy()

    #합성할 배경 이미지를 위의 파일 사이즈로 resize
    # 1/3눈  1/3입
    p_h = h
    p_y = int(y+h*5/6)
    p_w = int(w/10)
    p_x = x+int(p_w*1.05)
    n_w = int(w-p_w*1.7)
    if p_y + p_h > y+h:
        p_h = y+h-p_y
    resize_mask =  foreground.resize((n_w, int(p_h*1.9)))

    #이미지 합성
    background.paste(resize_mask, (p_x, p_y), resize_mask)

    #합성한 이미지 파일 보여주기
    save_file_name = img_id
    background.save(f'{save_dir}/{save_file_name}')
    
    for j in range(1,8):
        img_id = file_list[i+j]
        img_path = os.path.join(data_dir, img_id)
        img = Image.open(img_path)
        background = img.copy()
        background.paste(resize_mask, (p_x, p_y), resize_mask)
        save_file_name = img_id
        background.save(f'{save_dir}/{save_file_name}')