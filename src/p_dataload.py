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
    
        self.df = pd.read_csv(csv_path)
        assert self.df['image_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
"Some images referenced in the CSV file were not found"
        
        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X = self.df['image_name']
        self.y = self.mlb.fit_transform(self.df['tags'].str.split()).astype(np.float32)

    def X(self):
        return self.X
        
    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        label = from_numpy(self.y[index])
        return img, label

    def __len__(self):
        return len(self.df.index)
    
    def getLabelEncoder(self):
        return self.mlb
    
    def getDF(self):
        return self.df