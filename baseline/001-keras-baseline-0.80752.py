import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

import keras as k
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import MultiLabelBinarizer

import cv2
from tqdm import tqdm

RESOLUTION = 128
CACHE_FILE = '001-baseline-cache.h5'
THRESHOLD = 0.2

df_train = pd.read_csv('../data/train.csv')

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

split = 15000
x_train, x_valid, y_train, y_valid = X_train[:split], X_train[split:], y_train[:split], y_train[split:]

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(RESOLUTION,RESOLUTION, 3)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(17, activation='sigmoid'))

model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer='adam',
              metrics=['accuracy'])



if os.path.isfile(CACHE_FILE):
    print('####### Loading model from cache ######')
    model = load_model(CACHE_FILE)

else:
    print('####### Cache not found, building from scratch ######')
    model.fit(x_train, y_train,
              batch_size=64,
              epochs=6, # Should implement early stopping
              verbose=1,
              validation_data=(x_valid, y_valid))
    model.save(CACHE_FILE)

from sklearn.metrics import fbeta_score

p_valid = model.predict(x_valid, batch_size=128)
print(y_valid)
print(p_valid)
print(fbeta_score(y_valid, np.array(p_valid) > THRESHOLD, beta=2, average='samples'))


######## Prediction ########

df_test = pd.read_csv('../data/sample_submission.csv')

for file in tqdm(df_test['image_name'], miniters=256):
    img = cv2.imread('../data/test-jpg/{}.jpg'.format(file))
    X_test.append(cv2.resize(img,(RESOLUTION,RESOLUTION)))


X_test = np.array(X_test, np.float16) / 255.

y_pred = model.predict(X_test, batch_size=128)
# np.savetxt("pred-baseline.csv", y_pred, delimiter=";")

df_submission = pd.DataFrame()
df_submission['image_name'] = df_test['image_name']
df_submission['tags'] = [' '.join(x) for x in mlb.inverse_transform(y_pred > THRESHOLD)]

df_submission.to_csv('001-baseline.csv', index=False)
