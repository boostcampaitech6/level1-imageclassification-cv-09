from PIL import Image
import os
from tqdm import tqdm

folder_path = f'incorrect_image_gen\step0_aged_images' 
target_path = f'incorrect_image_gen\step1_resize'

if not os.path.exists(target_path):
    os.makedirs(target_path)

image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

for filename in tqdm(image_files, desc="Resizing Images"):
    img_path = os.path.join(folder_path, filename)
    save_path = os.path.join(target_path, filename)
    with Image.open(img_path) as img:
        img = img.resize((224, 224))
        img.save(save_path)
print("All images have been resized.")