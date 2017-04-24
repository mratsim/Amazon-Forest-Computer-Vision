from keras.applications.inception_v3 import InceptionV3

from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Input, BatchNormalization
from keras import optimizers
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import fbeta_score
from tqdm import tqdm
import cv2
import numpy as np
import os
from sklearn.preprocessing import MultiLabelBinarizer

RESOLUTION = 96
CACHE_FILE = '002-inception-baseline-cache.h5'
THRESHOLD = 0.2

def build_model():
    #Create own input format
    model_input = Input(shape=(RESOLUTION,RESOLUTION,3),name = 'image_input')
    
    #Load Inception v3
    base_model = InceptionV3(weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model(model_input)
    feat = Flatten(name='flatten')(x)
    feat = Dense(128, activation='relu')(feat)
    feat = BatchNormalization()(feat)
    out = Dense(17, activation='sigmoid')(feat)
    model = Model(inputs=model_input, outputs=out)
    
    model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
    
    
    print('######## Summary ########')
    model.summary()
    print('\n\n\n######## Config ########')
    model.get_config()
    print('\n\n\n######## ###### ########')
    
    return model

mlb = MultiLabelBinarizer()
X_train = []
X_test = []
df_train = pd.read_csv('../data/train.csv')
y_train = mlb.fit_transform(df_train['tags'].str.split())


for file in tqdm(df_train['image_name'], miniters=256):
    img = cv2.imread('../data/train-jpg/{}.jpg'.format(file))
    X_train.append(cv2.resize(img,(RESOLUTION,RESOLUTION)))

X_train = np.array(X_train, np.float16) / 255. ## TODO load per batch to avoid memory error here

print(X_train.shape)
print(y_train.shape)

######## Validation ########
x_trn, x_val, y_trn, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

if os.path.isfile(CACHE_FILE):
    print('####### Loading model from cache ######')
    model = load_model(CACHE_FILE)

else:
    print('####### Cache not found, building from scratch ######')
    model = build_model()
    model.fit(x_trn, y_trn,
              batch_size=64,
              epochs=15,
              verbose=1,
              validation_data=(x_val, y_val))
    model.save(CACHE_FILE)
    

p_valid = model.predict(x_val, batch_size=128)
print(y_val)
print(p_valid)
print(fbeta_score(y_val, np.array(p_valid) > THRESHOLD, beta=2, average='samples'))

######## Prediction ########

df_test = pd.read_csv('../data/sample_submission.csv')

for file in tqdm(df_test['image_name'], miniters=256):
    img = cv2.imread('../data/test-jpg/{}.jpg'.format(file))
    X_test.append(cv2.resize(img,(RESOLUTION,RESOLUTION)))


X_test = np.array(X_test, np.float16) / 255.

y_pred = model.predict(X_test, batch_size=128)

df_submission = pd.DataFrame()
df_submission['image_name'] = df_test['image_name']
df_submission['tags'] = [' '.join(x) for x in mlb.inverse_transform(y_pred > THRESHOLD)]

df_submission.to_csv('002-inception-baseline.csv', index=False)