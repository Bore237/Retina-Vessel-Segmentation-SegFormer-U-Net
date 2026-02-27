import os
from glob import glob
import numpy as np
import torch
import cv2
from monai.transforms import Compose, Lambdad, Resized, EnsureTyped
from monai.data import CacheDataset, DataLoader
from torch.utils.data import Dataset

def wrapper_load_grayscale_tiff(channel=1):
    """
    @brief Creates a loader function to read a specific channel from a TIFF image as a grayscale tensor.
    
    This wrapper allows you to select which channel (0=Red, 1=Green, 2=Blue) to extract from an RGB image.
    The resulting image tensor has shape (1, H, W) for a single channel.
    
    @param channel (int, optional): The channel index to load (0 to 2). Default is 1 (Green channel).
    
    @return: A function `load_grayscale_tiff(path)` that reads the TIFF image at `path` and returns a torch tensor.
    """
    
    def load_grayscale_tiff(path):
        """
        @brief Load a TIFF image and extract the specified channel as a torch tensor.
        
        @param path (str): Path to the TIFF image file.
        
        @return: torch.Tensor of shape (1, H, W) containing the selected channel as float32.
        
        @raises FileNotFoundError: If the image cannot be read.
        """
        # Read image without any change to original data
        img = np.array(cv2.imread(path, cv2.IMREAD_UNCHANGED))
        if img is None:
            raise FileNotFoundError(f"File not found or unreadable: {path}")
        
        # Convert BGR (OpenCV default) to RGB and reorder axes to (C, H, W)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32).transpose(2, 0, 1)

        # Extract the selected channel and keep as single-channel tensor
        if 0 <= channel <= 2:
            img = img[channel, :, :]
            img = np.expand_dims(img, axis=0)
            
        return torch.from_numpy(img)

    return load_grayscale_tiff

class ICRDataset:
    def __init__(
        self,
        image_dir,
        mask_dir,
        channel,
        img_size,
        apply_repeat=True,
        apply_clahe=True,
        batch_size=16,
        cache_rate=1.0,
        num_workers_cache=4,
        num_workers_loader=0,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.channel = channel
        self.img_size = img_size
        self.apply_repeat = apply_repeat
        self.apply_clahe = apply_clahe
        self.batch_size = batch_size
        self.cache_rate = cache_rate
        self.num_workers_cache = num_workers_cache
        self.num_workers_loader = num_workers_loader

        self.monai_files = self._transform_to_monai_format()
        self.dataset = self._build_dataset()

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    @staticmethod
    def repeat_channels_tensor(img):
        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.ndim == 3 and img.shape[0] == 1:
            pass
        else:
            raise ValueError(
                f"Expected tensor shape (H, W) or (1, H, W), got {img.shape}"
            )

        return img.repeat(3, 1, 1)

    @staticmethod
    def apply_CLAHE(img):
        if isinstance(img, torch.Tensor):
            img_np = img.detach().cpu().numpy()
        else:
            img_np = img

        if img_np.ndim == 3:
            img_np = img_np[0]

        img_min, img_max = img_np.min(), img_np.max()
        img_np = (img_np - img_min) / (img_max - img_min + 1e-8)
        img_np = (img_np * 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_np = clahe.apply(img_np)

        img_np = img_np.astype(np.float32) / 255.0
        img_np = np.expand_dims(img_np, axis=0)

        return torch.from_numpy(img_np)

    # --------------------------------------------------
    # Dataset preparation
    # --------------------------------------------------

    def _transform_to_monai_format(self):
        image_paths = sorted(glob(os.path.join(self.image_dir, "*")))
        mask_paths = sorted(glob(os.path.join(self.mask_dir, "*")))

        image_paths = [p.replace("\\", "/") for p in image_paths]
        mask_paths = [p.replace("\\", "/") for p in mask_paths]

        return [
            {"image": img, "label": msk}
            for img, msk in zip(image_paths, mask_paths)
        ]

    def _build_dataset(self):
        load_retina_img = wrapper_load_grayscale_tiff(self.channel)

        transforms_list = [
            Lambdad(keys=["image", "label"], func=load_retina_img),
        ]

        if self.apply_clahe:
            transforms_list.append(
                Lambdad(keys=["image"], func=self.apply_CLAHE)
            )

        transforms_list.extend([
            Resized(
                keys=["image", "label"],
                spatial_size=(self.img_size, self.img_size)
            ),
        ])

        if self.apply_repeat:
            transforms_list.append(
                Lambdad(
                    keys=["image"],
                    func=lambda x: self.repeat_channels_tensor(x)
                )
            )

        transforms_list.append(
            EnsureTyped(keys=["image", "label"])
        )

        transforms_cacheable = Compose(transforms_list)

        dataset = CacheDataset(
            data=self.monai_files,
            transform=transforms_cacheable,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers_cache
        )

        return dataset

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def get_dataset(self):
        return self.dataset

    def get_dataloader(self, transforms_random=None, shuffle=True):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers_loader,
            collate_fn=(
                (lambda batch: transforms_random(batch))
                if transforms_random
                else None
            )
        )


class HFMonaiWrapper(Dataset):
    def __init__(self, monai_dataset, feature_extractor, transforms=None):
        self.monai_dataset = monai_dataset
        self.feature_extractor = feature_extractor
        self.transforms = transforms

    def __len__(self):
        return len(self.monai_dataset)

    def __getitem__(self, idx):
        sample = self.monai_dataset[idx]

        # Apply MONAI transforms 
        if self.transforms:
            sample = self.transforms(sample)

        image = sample["image"]  
        label = sample["label"] 

        # Convert label to 0 or 1
        label = torch.where(label > 50, torch.tensor(1, dtype=torch.long), torch.tensor(0, dtype=torch.long)) 

        # SegFormer FeatureExtractor
        encoded = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze(0)

        return {"pixel_values": pixel_values, "labels": label}