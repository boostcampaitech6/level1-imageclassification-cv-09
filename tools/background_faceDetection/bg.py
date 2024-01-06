from facenet_pytorch import MTCNN
from rembg import remove
import cv2
import os
from PIL import Image
from tqdm import tqdm
import _thread
import threading
from multiprocessing import Pool
from time import time


def img_p(save_path, image_path, mtcnntemp):

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        margin = 40  # 여유 공간을 위한 마진 (픽셀 단위)

        # MTCNN을 사용하여 얼굴 검출 및 크롭

        boxes, _ = mtcnntemp.detect(image_rgb)
        if boxes is not None:
        # 첫 번째 검출된 얼굴의 영역을 추출하고 마진 추가
            box = boxes[0]
            x1, y1, x2, y2 = [max(0, int(coord)) for coord in [box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin]]
            face = image_rgb[y1:y2, x1:x2]

            # 배경 제거
            img_no_bg = remove(face)
            img_no_bg_pil = Image.fromarray(img_no_bg).convert("RGB")  # RGBA를 RGB로 변환
            img_no_bg_pil.save(save_path)
        
        else:
            # 얼굴이 검출되지 않은 경우, 원본 배경 제거
            img_no_bg = remove(image_rgb)
            img_no_bg_pil = Image.fromarray(img_no_bg).convert("RGB")  # RGBA를 RGB로 변환
            img_no_bg_pil.save(save_path)
            print(save_path)
            
        return True

if __name__ == "__main__" :
    p = Pool(8)
    
    img_dir = "/home/hojun/Documents/code/boostcamp_pr1/data/train/images"
    profiles = os.listdir(img_dir)
    profiles = [profile for profile in profiles if not profile.startswith(".")]
    profiles.sort()

    # MTCNN 객체 생성
    mtcnn = MTCNN( device='cuda:0')


    save_dir = "/home/hojun/Documents/code/boostcamp_pr1/remove_data2"
    os.makedirs(save_dir, exist_ok=True)  # 저장할 폴더 생성

    # profiles 리스트의 각 폴더에 대해 반복
    for profile in tqdm(profiles, desc="Processing profiles"):  # tqdm 적용
        # ... [프로필 폴더 처리 부분] ...
        folder_path = os.path.join(img_dir, profile)
        files = os.listdir(folder_path)
        files.sort()

        profile_save_dir = os.path.join(save_dir, profile)
        os.makedirs(profile_save_dir, exist_ok=True)  # 프로필별 저장 폴더 생성
        

    # profiles 리스트의 각 폴더에 대해 반복
    for profile in tqdm(profiles, desc="Processing profiles"):  # tqdm 적용
        # ... [프로필 폴더 처리 부분] ...
        folder_path = os.path.join(img_dir, profile)
        files = os.listdir(folder_path)
        files.sort()

        profile_save_dir = os.path.join(save_dir, profile)
        # os.makedirs(profile_save_dir, exist_ok=True)  # 프로필별 저장 폴더 생성
        
        for file in files:
            if file.startswith("."):
                continue
            image_path = os.path.join(folder_path, file)
            save_path = os.path.join(profile_save_dir, file)
            # print(save_path)
            # _thread.start_new_thread(img_p, (save_path, image_path))
            # thread = threading.Thread(target=img_p, args=(save_path, image_path, mtcnn))
            # thread.start()
            p.apply_async(img_p, (save_path, image_path, mtcnn))
            # img_p(save_path, image_path)

    while 1:
        pass

    print("이미지 처리 및 저장이 완료되었습니다.")
