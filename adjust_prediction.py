import numpy as np
import os
import pandas as pd
import torch
from src.p_dataload import KaggleAmazonDataset


## Load MultiLabelBinarizer config
X_train = KaggleAmazonDataset('./data/train.csv','./data/train-jpg/','.jpg')
mlb = X_train.getLabelEncoder()

## Load sample submission:
df_test = pd.read_csv('./data/sample_submission_v2.csv')

## Load raw prediction (proba):
subm_proba = np.loadtxt('./out/2017-05-12_1223-resnet50-L2reg-new-data-raw-pred-0.922374050536.csv',
                       delimiter=';')

## Load threshold:
model_path = './snapshots/2017-05-12_1223-resnet50-L2reg-new-data-model_best.pth'
checkpoint = torch.load(model_path)
threshold = checkpoint['threshold']

## Force single weather: TODO check if cloudy is alone
weather = subm_proba[:, 0:4]
indices = np.argmax(weather, axis=1)
new_weather = np.eye(4)[indices]
subm_proba[:,0:4] = new_weather

predictions = subm_proba > threshold

result = pd.DataFrame({
    'image_name': df_test['image_name'],
    'tags': mlb.inverse_transform(predictions)
    })
result['tags'] = result['tags'].apply(lambda tags: " ".join(tags))
    
result_path = './out/2017-05-12_1223-resnet50-L2reg-new-data-adjusted-pred-0.922374050536.csv'
result.to_csv(result_path, index=False)