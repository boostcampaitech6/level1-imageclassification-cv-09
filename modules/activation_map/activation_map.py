from PIL import Image
from torchvision import transforms
from torchcam.methods import SmoothGradCAMpp
from model.models import get_model
import os
import torch
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.io.image import read_image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([224,224]),
    transforms.Normalize(mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246))
    ])

img = read_image("/home/hojun/Documents/code/boostcamp_pr1/data/data_base/train/images/000001_female_Asian_45/mask1.jpg")
base_img = img
img = normalize(resize(img, (224, 224)) / 255., [0.548, 0.504, 0.479], [0.237, 0.247, 0.246])
model = get_model("tiny_vit_21m_224_dist_in22k_ft_in1k_froze")
model = model(num_classes=18)

check_point_path = os.path.join("/home/hojun/Documents/code/boostcamp_pr1/results/train/20231221_072123/best_model.pt")
check_point = torch.load(check_point_path,map_location=torch.device("cpu"))
model.load_state_dict(check_point['model'])

with SmoothGradCAMpp(model) as cam_extractor:
  # Preprocess your data and feed it to the model
  out = model(img.unsqueeze(0))
  # Retrieve the CAM by passing the class index and the model output
  activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
result = overlay_mask(to_pil_image(base_img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
# Display it
plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()