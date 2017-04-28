import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

RESOLUTION = 96 # Ideally we shouldn't be resizing but I'm lacking memory

if __name__ == "__main__":
    data = []
    df_train = pd.read_csv('./data/train.csv')

    for file in tqdm(df_train['image_name'], miniters=256):
        img = cv2.imread('./data/train-jpg/{}.jpg'.format(file))
        data.append(cv2.resize(img,(RESOLUTION,RESOLUTION)))

    data = np.array(data, np.float32) / 255 # Must use float32 at least otherwise we get over float16 limits
    print("Shape: ", data.shape)

    means = []
    stdevs = []
    for i in range(3):
        pixels = data[:,:,:,i].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    print("means: {}".format(means))
    print("stdevs: {}".format(stdevs))
    print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))