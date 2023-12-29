import torch
from torch.utils.data import Dataset, Subset, random_split, WeightedRandomSampler
from torchvision.transforms import (
    Resize,
    ToTensor,
    Normalize,
    Compose,
    CenterCrop,
    ColorJitter,
)
import numpy as np
from collections import defaultdict
from typing import Tuple, List
from enum import Enum
from PIL import Image
import os,random

from sklearn.model_selection import train_test_split
import pandas as pd

def get_dataset_function(dataset_function_str):
    if dataset_function_str == "baseDataset":
        return MaskBaseDataset
    elif dataset_function_str == "SplitByProfileDataset_weightSampler_stratify":
        return MaskSplitByProfileDataset


# 지원되는 이미지 확장자 리스트
IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    """파일 이름이 이미지 확장자를 가지는지 확인하는 함수

    Args:
        filename (str): 확인하고자 하는 파일 이름

    Returns:
        bool: 파일 이름이 이미지 확장자를 가지면 True, 그렇지 않으면 False.
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



class MaskLabels(int, Enum):
    """마스크 라벨을 나타내는 Enum 클래스"""

    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    """성별 라벨을 나타내는 Enum 클래스"""

    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        """문자열로부터 해당하는 성별 라벨을 찾아 반환하는 클래스 메서드"""
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(
                f"Gender value should be either 'male' or 'female', {value}"
            )


class AgeLabels(int, Enum):
    """나이 라벨을 나타내는 Enum 클래스"""

    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        """숫자로부터 해당하는 나이 라벨을 찾아 반환하는 클래스 메서드"""
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskBaseDataset(Dataset):
    """마스크 데이터셋의 기본 클래스"""

    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
    }

    image_paths = []
    mask_labels = []
    gender_labels = [] 
    
    age_labels = []

    def __init__(
        self,
        data_dir,
        transform=None,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio = 0.2
    ):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = transform
        self.setup()  # 데이터셋을 설정
        self.calc_statistics()  # 통계시 계산 (평균 및 표준 편차)

    def setup(self):
        """데이터 디렉토리로부터 이미지 경로와 라벨을 설정하는 메서드"""
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if (
                    _file_name not in self._file_names
                ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(
                    self.data_dir, profile, file_name
                )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        """데이터셋의 통계치를 계산하는 메서드"""
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print(
                "[Warning] Calculating statistics... It can take a long time depending on your CPU machine"
            )
            sums = []
            squared = []
            for image_path in self.image_paths[:]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image**2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean**2) ** 0.5 / 255

    def set_transform(self, transform):
        """변환(transform)을 설정하는 메서드"""
        self.transform = transform

    def __getitem__(self, index):
        """인덱스에 해당하는 데이터를 가져오는 메서드"""
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        """데이터셋의 길이를 반환하는 메서드"""
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        """인덱스에 해당하는 마스크 라벨을 반환하는 메서드"""
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        """인덱스에 해당하는 성별 라벨을 반환하는 메서드"""
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        """인덱스에 해당하는 나이 라벨을 반환하는 메서드"""
        return self.age_labels[index]

    def read_image(self, index):
        """인덱스에 해당하는 이미지를 읽는 메서드"""
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        """다중 라벨을 하나의 클래스로 인코딩하는 메서드"""
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(
        multi_class_label,
    ) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        """인코딩된 다중 라벨을 각각의 라벨로 디코딩하는 메서드"""
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        """정규화된 이미지를 원래대로 되돌리는 메서드"""
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """데이터셋을 학습과 검증용으로 나누는 메서드
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여 torch.utils.data.Subset 클래스 둘로 나눕니다.
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set

    """마스크 데이터셋의 기본 클래스"""

    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(
        self,
        data_gen_dir,
        transform=None,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio = 0.2,
    ):
        self.data_dir = data_gen_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = transform
        self.setup()  # 데이터셋을 설정
        self.calc_statistics()  # 통계시 계산 (평균 및 표준 편차)

    def setup(self):
        """데이터 디렉토리로부터 이미지 경로와 라벨을 설정하는 메서드"""
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_path = os.path.join(self.data_dir, profile)
            _file_name, ext = os.path.splitext(profile)
            if _file_name != "generated":  # Adjust based on your filename structure
                continue

            if "original" in profile:
                continue

            mask_label = self._file_names[_file_name]

            # Extract information from the filename
            id, gender, race, origin_age, _, age, _ = profile.split("_")
            gender_label = GenderLabels.from_str(gender)
            age_label = AgeLabels.from_number(age)

            self.image_paths.append(img_path)
            self.mask_labels.append(mask_label)
            self.gender_labels.append(gender_label)
            self.age_labels.append(age_label)

    def calc_statistics(self):
        """데이터셋의 통계치를 계산하는 메서드"""
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print(
                "[Warning] Calculating statistics... It can take a long time depending on your CPU machine"
            )
            sums = []
            squared = []
            for image_path in self.image_paths[:]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image**2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean**2) ** 0.5 / 255

    def set_transform(self, transform):
        """변환(transform)을 설정하는 메서드"""
        self.transform = transform

    def __getitem__(self, index):
        """인덱스에 해당하는 데이터를 가져오는 메서드"""
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        """데이터셋의 길이를 반환하는 메서드"""
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        """인덱스에 해당하는 마스크 라벨을 반환하는 메서드"""
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        """인덱스에 해당하는 성별 라벨을 반환하는 메서드"""
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        """인덱스에 해당하는 나이 라벨을 반환하는 메서드"""
        return self.age_labels[index]

    def read_image(self, index):
        """인덱스에 해당하는 이미지를 읽는 메서드"""
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        """다중 라벨을 하나의 클래스로 인코딩하는 메서드"""
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(
        multi_class_label,
    ) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        """인코딩된 다중 라벨을 각각의 라벨로 디코딩하는 메서드"""
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        """정규화된 이미지를 원래대로 되돌리는 메서드"""
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """데이터셋을 학습과 검증용으로 나누는 메서드
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여 torch.utils.data.Subset 클래스 둘로 나눕니다.
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
    train / val 나누는 기준을 이미지에 대해서 random 이 아닌 사람(profile)을 기준으로 나눕니다.
    구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다.
    이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
        seed = 42,
        drop_age_mode = ["train", "val"],
        drop_age = ["57", "58", "59"]
    ):
        self.indices = defaultdict(list)
        self.seed = seed
        self.drop_age_mode  = drop_age_mode
        self.drop_age = drop_age
        super().__init__(data_dir, mean, std, val_ratio)
    

    @staticmethod
    def _split_profile(profiles, val_ratio):
        """프로필을 학습과 검증용으로 나누는 메서드"""
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.sample(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {"train": train_indices, "val": val_indices}


    # train set과 valid set을 비슷한 클래스 비율로 나눔
    def balanced_split_profile(self, profiles, val_ratio):
        df = pd.DataFrame()
        
        df["path"] = profiles
        
        multi_labels = []
        gender_labels = []
        age_labels = []
        
        for profile in profiles:
            id, gender, race, age = profile.split("_")
            gender_label = GenderLabels.from_str(gender)
            age_label = AgeLabels.from_number(age)

            gender_labels.append(gender_label)
            age_labels.append(age_label)

        for i in range(len(profiles)):
            multi_labels.append(gender_labels[i] * 3 + age_labels[i])
            
        df["multi_label"] = multi_labels
        df["gender_label"] = gender_labels
        df["age_label"] = age_labels
                
        train, val =  train_test_split(
            df,
            test_size=val_ratio,
            random_state=self.seed,
            stratify=df["multi_label"]
        )
        
        train_indices = set(list(train.index))
        val_indices = set(list(val.index))

        return {"train": train_indices, "val": val_indices}
    
    
    # age_remove
    def age_drop(
        self,
        split_profiles,
        profiles,
        mode=["train", "val"],
        age_list=["57", "58", "59"],
    ):  # mode에서 특정 나이 제거
        for phase, indices in split_profiles.items():
            removed_idx = []
            if phase in mode:
                for idx in indices:
                    if profiles[idx].split("_")[-1] in age_list:
                        removed_idx.append(idx)
            for ri in removed_idx:
                split_profiles[phase].remove(ri)

        return split_profiles
                
        
    def setup(self):
        """데이터셋 설정을 하는 메서드. 프로필 기준으로 나눈다."""
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        # split_profiles = self._split_profile(profiles, self.val_ratio)
        split_profiles = self.balanced_split_profile(profiles, self.val_ratio)
        
        split_profiles = self.age_drop(
            split_profiles, profiles, self.drop_age_mode, self.drop_age
        )

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if (
                        _file_name not in self._file_names
                    ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(
                        self.data_dir, profile, file_name
                    )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset]:
        """프로필 기준으로 나눈 데이터셋을 Subset 리스트로 반환하는 메서드"""
        return [Subset(self, indices) for phase, indices in self.indices.items()]
    
    
    # Data Imbalance를 해결하기 위해 WeightedRandomSampler 사용
    def get_sampler(self, phase) :
        multi_class = []
        for phase_idx in self.indices[phase]:
            temp = self.encode_multi_class(self.mask_labels[phase_idx],
                                    self.gender_labels[phase_idx],
                                    self.age_labels[phase_idx])
            multi_class.append(temp)
       
        class_sample_count = np.array([len(np.where(multi_class == t)[0]) for t in np.unique(multi_class)])		   
        weight = 1. / class_sample_count
								  
        samples_weight = np.array([weight[t] for t in multi_class])
        samples_weight = torch.from_numpy(samples_weight).double()
        phase_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        return phase_sampler


class TestDataset(Dataset):
    """테스트 데이터셋 클래스"""

    def __init__(
        self, img_paths, transform
    ):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        """인덱스에 해당하는 데이터를 가져오는 메서드"""
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        """데이터셋의 길이를 반환하는 메서드"""
        return len(self.img_paths)
