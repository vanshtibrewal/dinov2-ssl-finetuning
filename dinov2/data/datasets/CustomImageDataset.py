import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from .decoders import TargetDecoder, ImageDataDecoder

class CustomImageDataset(Dataset):
    def __init__(
            self,
            split,
            root: str,
            extra: str,
            transform=None,
            target_transform=None,
            test_size=0.2,
            random_state=42):
        self.img_labels = pd.read_csv(extra)
        self.img_dir = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split

        # split into train and test
        self.train_data, self.test_data = train_test_split(
            self.img_labels, test_size=test_size, random_state=random_state)

    def __len__(self):
        return len(self.train_data) 

    def __getitem__(self, idx):
        # read image
        try:
            img_path = os.path.join(self.img_dir, self.train_data.iloc[idx, 0])
            with open(img_path, mode="rb") as f:
                image_pil = f.read()
            #image = read_image(img_path)
            #image_pil = transforms.ToPILImage()(image)
            image_pil = ImageDataDecoder(image_pil).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {idx}") from e
        if self.transform:
            image_pil = self.transform(image_pil)

        return image_pil, None

    def get_test_item(self, idx):
        img_path = os.path.join(self.img_dir, self.test_data.iloc[idx, 0])
        image = read_image(img_path)
        image_pil = transforms.ToPILImage()(image)
        if self.transform:
            image_pil = self.transform(image_pil)
        return image_pil

    
