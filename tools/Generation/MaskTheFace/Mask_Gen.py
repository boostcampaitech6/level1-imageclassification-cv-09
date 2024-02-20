import os
import cv2
import dlib
import numpy as np
from tqdm import tqdm
from utils.aux_functions import *
import argparse
import re

def create_masked_images(image_path, args, output_folder):
    if is_image(image_path):
        # 마스크 생성
        masked_image, mask, _, _ = mask_image(image_path, args)
        for i, img in enumerate(masked_image):
            # 저장할 파일명과 경로를 설정
            masked_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_{mask[i]}.jpg")
            cv2.imwrite(masked_file_path, img)

def main():
    args = argparse.Namespace()
    args.detector = dlib.get_frontal_face_detector()
    path_to_dlib_model = '../MaskTheFace/dlib_models/shape_predictor_68_face_landmarks.dat'
    args.predictor = dlib.shape_predictor(path_to_dlib_model)

    args.mask_type = "surgical"  # 사용할 마스크 타입
    args.pattern = ""           # 패턴
    args.pattern_weight = 0.5   # 패턴 가중치
    args.color = "#0473e2"      # 마스크 색상
    args.color_weight = 0.5     # 색상 가중치
    args.code = ""              # 특정 포맷 생성
    args.verbose = False        # 상세 출력 여부
    args.write_original_image = False # 원본 이미지 저장 여부

    parent_folder = "../MaskTheFace/masked"

    # 마스크 이미지를 저장할 폴더 이름 생성
    output_folder_name = f"{args.mask_type}_{args.pattern}_{args.pattern_weight}_{args.color}_{args.color_weight}"
    output_folder_path = os.path.join(parent_folder, output_folder_name)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # 현재 폴더 내의 모든 이미지에 대해 처리
    for filename in tqdm(os.listdir(parent_folder), desc="Processing images"):
        file_path = os.path.join(parent_folder, filename)
        create_masked_images(file_path, args, output_folder_path)

if __name__ == "__main__":
    main()
