from torchvision import transforms
from facenet_pytorch import MTCNN
from rembg import remove
from PIL import Image
import numpy as np
# import mediapipe as mp
import cv2

def get_transform_function(transform_function_str,config):
    
    if transform_function_str == "baseTransform":
        return baseTransform(config)
    elif transform_function_str == "centerCrop_transform":
        return CCTransform(config)
    elif transform_function_str == "faceCrop_transform":
        return FCTransform(config)


# class FaceCropAndRemoveBG(object):
#     def __init__(self, config, margin=30):
#         self.mtcnn = MTCNN()
#         self.margin = margin
#         self.center_crop_size = config['centor_crop']

#     def __call__(self, img):
#         # 필요한 경우 RGB로 변환
#         img = img.convert('RGB')

#         # 얼굴 검출 및 크롭
#         boxes, _ = self.mtcnn.detect(img)
#         if boxes is not None:
#             box = boxes[0]
#             width, height = img.size

#             # 여백을 추가하여 크롭 영역 계산
#             x1, y1, x2, y2 = box
#             x1 = max(0, x1 - self.margin)
#             y1 = max(0, y1 - self.margin)
#             x2 = min(width, x2 + self.margin)
#             y2 = min(height, y2 + self.margin)

#             img_cropped = img.crop((x1, y1, x2, y2))

#             # 배경 제거
#             img_no_bg = remove(np.array(img_cropped))
#             img_no_bg_pil = Image.fromarray(img_no_bg)

#             # RGBA 이미지를 RGB로 변환
#             if img_no_bg_pil.mode == 'RGBA':
#                 img_no_bg_pil = img_no_bg_pil.convert('RGB')

#             return img_no_bg_pil
#         else:
#             # 얼굴 검출 실패 시 중앙 크롭 수행
#             center_crop = transforms.CenterCrop(self.center_crop_size)
#             return center_crop(img)


class FaceCropAndRemoveBG(object):
    def __init__(self, config, margin=40):
        self.mtcnn = MTCNN()
        self.margin = margin
        self.center_crop_size = config['centor_crop']

    def __call__(self, img):

        # 얼굴 검출 및 크롭
        boxes, _ = self.mtcnn.detect(img)
        if boxes is not None:
            box = boxes[0]
            width, height = img.size

            # 여백을 추가하여 크롭 영역 계산
            x1, y1, x2, y2 = box
            x1 = max(0, x1 - self.margin)
            y1 = max(0, y1 - self.margin)
            x2 = min(width, x2 + self.margin)
            y2 = min(height, y2 + self.margin)

            img_cropped = img.crop((x1, y1, x2, y2))
            
            return img_cropped
        else:
            # 얼굴 검출 실패 시 중앙 크롭 수행
            center_crop = transforms.CenterCrop(self.center_crop_size)
            return center_crop(img)
        
        
# class FaceCropAndRemoveBG(object):
#     def __init__(self, config, margin=30):
#         self.mtcnn = MTCNN()
#         self.margin = margin
#         self.center_crop_size = config['centor_crop']
#         self.mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

#     def remove_background(self, img):
#         results = self.mp_selfie_segmentation.process(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
#         condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
#         output_image = np.where(condition, np.array(img), np.zeros(img.shape, dtype=np.uint8))
#         return output_image

#     def __call__(self, img):
#         img = img.convert('RGB')

#         # 얼굴 검출 및 크롭
#         boxes, _ = self.mtcnn.detect(img)
#         if boxes is not None:
#             box = boxes[0]
#             width, height = img.size
#             x1, y1, x2, y2 = box
#             x1 = max(0, x1 - self.margin)
#             y1 = max(0, y1 - self.margin)
#             x2 = min(width, x2 + self.margin)
#             y2 = min(height, y2 + self.margin)

#             img_cropped = img.crop((x1, y1, x2, y2))

#             # 배경 제거
#             img_no_bg = self.remove_background(img_cropped)
#             img_no_bg_pil = Image.fromarray(img_no_bg)

#             return img_no_bg_pil
#         else:
#             # 얼굴 검출 실패 시 중앙 크롭 수행
#             center_crop = transforms.CenterCrop(self.center_crop_size)
#             return center_crop(img)
        

def baseTransform(config):
    return transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(config['resize_size']),
    transforms.Normalize(mean=config['mean'],
                        std=config['std'])
    ])
    
def CCTransform(config):
    return transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(config['centor_crop']),
    transforms.Resize(config['resize_size']),
    transforms.Normalize(mean=config['mean'],
                        std=config['std'])
    ])
    
    
def FCTransform(config):
    return transforms.Compose([
    FaceCropAndRemoveBG(config),
    transforms.ToTensor(),
    transforms.Resize(config['resize_size']),
    transforms.Normalize(mean=config['mean'],
                        std=config['std'])        
    ])