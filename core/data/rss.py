import os
import torch
import numpy as np
from copy import deepcopy
import cv2
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, random_split

from torchvision.transforms.transforms import ToPILImage
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A


def transforms_v1(crop=(384, 384)):
    train_transforms = A.Compose(
        [
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0.1, rotate_limit=25, interpolation=cv2.INTER_AREA, p=0.75),
            A.RandomCrop(*crop),
            A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, p=0.25),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.GaussianBlur(sigma_limit=0.7, p=0.5),
            A.GaussNoise(mean=10, p=0.5),
            # A.ToFloat(max_value=255),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ],
        additional_targets={"jmask": "mask"},
    )

    test_transforms = A.Compose(
        [
            A.RandomCrop(*crop),
            # A.ToFloat(max_value=255),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ],
        additional_targets={"jmask": "mask"},
    )

    return train_transforms, test_transforms


class RoadSegDataset(Dataset):
    def __init__(self, filepaths, test=False, transform=None):
        self.filepaths = filepaths
        self.test = test
        self.transform = transform

    @classmethod
    def from_directory(cls, image_dir, mask_dir, **kwargs):
        filepaths = cls._get_images_filepaths(image_dir, mask_dir, ignore_non_matching=False)
        return cls(filepaths=filepaths, **kwargs)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        if self.test:
            image_fp = self.filepaths[idx]
            image = np.array(Image.open(image_fp).convert("RGB"))
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]
            return image

        image_fp, mask_fp = self.filepaths[idx]
        image = np.array(Image.open(image_fp).convert("RGB"))
        mask = np.array(Image.open(mask_fp).convert("L")) / 255

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask

    @staticmethod
    def _get_images_filepaths(image_dir, mask_dir=None, ignore_non_matching=False):
        """Find all the images from the images_dir directory and
        the segmentation images from the mask_dir directory
        while checking integrity of data"""

        ACCEPTABLE_IMAGE_FORMATS = ["jpg", "jpeg", "png"]
        ACCEPTABLE_SEGMENTATION_FORMATS = ["png"]

        image_filepaths = sum(
            [list(Path(image_dir).rglob("*." + ext)) for ext in ACCEPTABLE_IMAGE_FORMATS],
            [],
        )
        if mask_dir is None:
            return image_filepaths

        mask_filepaths = sum(
            [list(Path(mask_dir).rglob("*." + ext)) for ext in ACCEPTABLE_SEGMENTATION_FORMATS],
            [],
        )

        mask_name2fp = {p.stem: p for p in mask_filepaths}
        pairs = []

        for image in image_filepaths:
            if image.stem not in mask_name2fp:
                if ignore_non_matching is False:
                    raise RuntimeError("No corresponding segmentation found for image {0}.".format(image))
                continue
            mask = mask_name2fp[image.stem]
            pairs.append((image, mask))

        return pairs


class RSS_MultiMask(RoadSegDataset):
    def __getitem__(self, idx):
        if self.test:
            image_fp = self.filepaths[idx]
            image = np.array(Image.open(image_fp).convert("RGB"))
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]
            return image

        image_fp, mask_fp = self.filepaths[idx]
        mask_fname, ext = os.path.splitext(mask_fp)
        jmask_fp = mask_fname + "_junction" + ext
        image = np.array(Image.open(image_fp).convert("RGB"))
        mask = np.array(Image.open(mask_fp).convert("L")) / 255
        jmask = np.array(Image.open(jmask_fp).convert("L")) / 255

        if self.transform:
            transformed = self.transform(image=image, mask=mask, jmask=jmask)
            image = transformed["image"]
            mask = transformed["mask"]
            jmask = transformed["jmask"]

        mask = np.stack((1 - mask, mask), axis=0)
        jmask = np.stack((1 - jmask, jmask), axis=0)
        return image, (mask, jmask)