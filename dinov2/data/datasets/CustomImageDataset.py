import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
#from sklearn.model_selection import train_test_split
from .decoders import ImageDataDecoder
#import numpy as np
#from PIL import Image

def save_image(tensor, name1, name2):
    # Convert the tensor to a NumPy array
    #numpy_array = tensor.cpu().numpy()
    
    # Convert to uint8 and scale to [0, 255]
    #numpy_array = numpy_array
    
    # Create an image from the NumPy array
    #image = Image.fromarray(numpy_array.transpose(1, 2, 0))  # Convert to HWC format
    
    # Concatenate name1 and name2 to form the file name
    file_name = f'{name1}_{name2}.png'
    
    # Specify the save path with the concatenated file name
    save_path = f'/home/ubuntu/example_image/augmented_images/{file_name}'
    
    # Save the image as a PNG file
    #image.save(save_path)
    # to make it work, comment out normalization in augmentations
    tensor.save(save_path)

def save_all(image_pil, name1):
    # save both global crops
    save_image(image_pil['global_crops'][0], name1, 'global_crop1')
    save_image(image_pil['global_crops'][1], name1, 'global_crop2')

    #save local crops
    local_crops = image_pil['local_crops']
    for i, image in enumerate(local_crops):
        save_image(image, name1, 'local_crop'+str(i))


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
        #self.train_data, self.test_data = train_test_split(
        #    self.img_labels, test_size=test_size, random_state=random_state)

    def __len__(self):
        return len(self.img_labels) 
        #return len(self.train_data) 

    def __getitem__(self, idx):
        # read image
        try:
            #img_path = os.path.join(self.img_dir, self.train_data.iloc[idx, 0])
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            with open(img_path, mode="rb") as f:
                image_pil = f.read()
            #image = read_image(img_path)
            #image_pil = transforms.ToPILImage()(image)
            image_pil = ImageDataDecoder(image_pil).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {idx}") from e
        if self.transform:
            image_pil = self.transform(image_pil)

        # safe images, uncomment line below to save images
        #save_all(image_pil, self.train_data.iloc[idx, 1])

        return image_pil, None

    def get_test_item(self, idx):
        img_path = os.path.join(self.img_dir, self.test_data.iloc[idx, 0])
        image = read_image(img_path)
        image_pil = transforms.ToPILImage()(image)
        if self.transform:
            image_pil = self.transform(image_pil)
        return image_pil
    
    
