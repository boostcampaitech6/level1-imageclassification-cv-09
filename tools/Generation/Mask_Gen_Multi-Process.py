import os
import cv2
import dlib
import numpy as np
from tqdm import tqdm
from utils.aux_functions import *
import argparse
import re
from multiprocessing import Pool

def create_masked_images(image_path, args, output_folder):
    if is_image(image_path):
        masked_image, mask, _, _ = mask_image(image_path, args)
        for i, img in enumerate(masked_image):
            masked_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_{mask[i]}.jpg")
            cv2.imwrite(masked_file_path, img)

def process_image(file_path, args, output_folder):
    create_masked_images(file_path, args, output_folder)

def main():
    args = argparse.Namespace()
    args.detector = dlib.get_frontal_face_detector()
    path_to_dlib_model = 'C:/boostcamp_ai/Project_1/MaskTheFace/dlib_models/shape_predictor_68_face_landmarks.dat'
    args.predictor = dlib.shape_predictor(path_to_dlib_model)


    # 각 종류 별 color 3가지 = 3*4 = 12
    # 흰색, 하늘색("#0473e2"), 회색
    # pattern 3종류 = 3 * 3 = 9 
    # => 총 21가지 마스크 이미지 생성

    
    args.mask_type = "surgical"
    # args.mask_type = "cloth"
    # args.mask_type = "N95"
    # args.mask_type = "KN95"
    args.pattern = ""
    args.pattern_weight = 0.7
    # 흰색
    args.color = "#0473e2"
    # 회색
    # args.color = "#5f4e4e"
    # 분홍색
    # args.color = "#9e779c"
    
    args.color_weight = 0.7
    args.code = ""
    args.verbose = False
    args.write_original_image = False

    # parent_folder = 'D:/231217_generated_aligned_filtered_512x512'
    # parent_folder = 'D:/231217_generated_aligned_filtered_512x512'
    # parent_folder = "D:/231220_generated_aligned_croped/231220_generated_aligned"
    parent_folder = "C:/Users/bjong/OneDrive/바탕 화면/masked"

    output_folder_name = f"{args.mask_type}_{args.pattern}_{args.pattern_weight}_{args.color}_{args.color_weight}"
    output_folder_path = os.path.join(parent_folder, output_folder_name)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    file_paths = [(os.path.join(parent_folder, filename), args, output_folder_path) for filename in os.listdir(parent_folder)]

    with Pool() as pool:
        list(tqdm(pool.starmap(process_image, file_paths), total=len(file_paths), desc="Processing images"))

if __name__ == "__main__":
    main()
