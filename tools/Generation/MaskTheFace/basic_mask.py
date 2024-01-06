import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import torchvision

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def detect_faces_and_mask(image_path, model):
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image_tensor)

    if len(predictions[0]["boxes"]) > 0:
        box = predictions[0]["boxes"][0].cpu().numpy()
        mask_image(image, box)

def mask_image(image, box, mask_path):
    """
    image: PIL Image
    box: 얼굴 영역의 bounding box, [x1, y1, x2, y2] 형식
    mask_path: 마스크 이미지 파일 경로
    """
    mask_path = 'C:/boostcamp_ai/Project_1/MaskTheFace/masks/templates/N95.png'
    # 마스크 이미지 불러오기
    mask = Image.open(mask_path)
    
    # 얼굴 영역 크기에 맞게 마스크 이미지 조정
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    mask = mask.resize((int(box_width), int(box_height)))

    # 마스크 이미지를 얼굴 영역에 합성
    image.paste(mask, (int(box[0]), int(box[1])), mask)

    return image

# 얼굴 검출 모델 불러오기
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device).eval()

# 이미지 파일 경로
image_path = "C:/Users/bjong/Downloads/231217_generated_aligned-20231217T224558Z-001/231217_generated_aligned/000002_female_Asian_52_generated_50_aligned/000002_female_Asian_52_generated_50_aligned.jpg"

# 얼굴 검출 및 마스킹
detect_faces_and_mask(image_path, model)
