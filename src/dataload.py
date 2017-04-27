from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import os
from PIL import Image # Replace by accimage when ready
from PIL.Image import FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM, ROTATE_90, ROTATE_180, ROTATE_270
from PIL.ImageEnhance import Color, Contrast, Brightness, Sharpness
from sklearn.preprocessing import MultiLabelBinarizer
from torch import np, from_numpy # Numpy like wrapper

class KaggleAmazonDataset(Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
    """

    def __init__(self, csv_path, img_path, img_ext, transform=None):
    
        tmp_df = pd.read_csv(csv_path)
        assert tmp_df['image_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
"Some images referenced in the CSV file were not found"
        
        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df['image_name']
        self.y_train = self.mlb.fit_transform(tmp_df['tags'].str.split()).astype(np.float32)

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        label = from_numpy(self.y_train[index])
        return img, label

    def __len__(self):
        return len(self.X_train.index)

class AugmentedAmazonDataset(Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.
    This dataset is augmented

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
    """

    def __init__(self, csv_path, img_path, img_ext, transform=None):
    
        tmp_df = pd.read_csv(csv_path)
        assert tmp_df['image_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
"Some images referenced in the CSV file were not found"
        
        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df['image_name']
        self.y_train = self.mlb.fit_transform(tmp_df['tags'].str.split()).astype(np.float32)
        self.augmentNumber = 14 # TODO, do something about this harcoded value

    def __getitem__(self, index):
        real_length = self.real_length()
        real_index = index % real_length
        
        img = Image.open(self.img_path + self.X_train[real_index] + self.img_ext)
        img = img.convert('RGB')
        
        ## Augmentation code
        if 0 <= index < real_length:
            pass
        
        ### Mirroring and Rotating
        elif real_length <= index < 2 * real_length:
            img = img.transpose(FLIP_LEFT_RIGHT)
        elif 2 * real_length <= index < 3 * real_length:
            img = img.transpose(FLIP_TOP_BOTTOM)
        elif 3 * real_length <= index < 4 * real_length:
            img = img.transpose(ROTATE_90)
        elif 4 * real_length <= index < 5 * real_length:
            img = img.transpose(ROTATE_180)
        elif 5 * real_length <= index < 6 * real_length:
            img = img.transpose(ROTATE_270)

        ### Color balance
        elif 6 * real_length <= index < 7 * real_length:
            img = Color(img).enhance(0.95)
        elif 7 * real_length <= index < 8 * real_length:
            img = Color(img).enhance(1.05)
        ## Contrast
        elif 8 * real_length <= index < 9 * real_length:
            img = Contrast(img).enhance(0.95)
        elif 9 * real_length <= index < 10 * real_length:
            img = Contrast(img).enhance(1.05)
        ## Brightness
        elif 10 * real_length <= index < 11 * real_length:
            img = Brightness(img).enhance(0.95)
        elif 11 * real_length <= index < 12 * real_length:
            img = Brightness(img).enhance(1.05)
        ## Sharpness
        elif 12 * real_length <= index < 13 * real_length:
            img = Sharpness(img).enhance(0.95)
        elif 13 * real_length <= index < 14 * real_length:
            img = Sharpness(img).enhance(1.05)
        else:
            raise IndexError("Index out of bounds")
            
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = from_numpy(self.y_train[real_index])
        return img, label
    
    def __len__(self):
        return len(self.X_train.index) * self.augmentNumber
    
    def real_length(self):
        return len(self.X_train.index)