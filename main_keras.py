import numpy as np
import pandas as pd

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D

from timeit import default_timer as timer
from src.k_dataloader import AmazonGenerator
from src.k_model_selection import train_valid_split

from sklearn.metrics import fbeta_score

RESOLUTION = 256

if __name__ == "__main__":
    # Initiate timer
    global_timer = timer()

    # Setting random seeds for reproducibility. (Caveat, some CuDNN algorithms are non-deterministic)
    np.random.seed(1337)    
    
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
    model.add(Dense(96, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(17, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    train_gen = AmazonGenerator(featurewise_center=True,
                            featurewise_std_normalization=True,
                            width_shift_range=0.15,
                            horizontal_flip=True,
                            rotation_range=15,
                            rescale=1./255
                           )
    
    valid_gen = AmazonGenerator(featurewise_center=True,
                                featurewise_std_normalization=True,
                                rescale=1./255)
    
    # train_gen.fit_from_csv('./data/train.csv',
    #                               './data/train-jpg/',
    #                               '.jpg',
    #                              rescale=1./255,
    #                              target_size=(RESOLUTION,RESOLUTION))
    
    # train_gen.dump_dataset_mean_std('train_256_mean.npy', 'train_256_std.npy')
    train_gen.load_mean_std('train_256_mean.npy', 'train_256_std.npy')
    valid_gen.load_mean_std('train_256_mean.npy', 'train_256_std.npy')
    
    df_train = pd.read_csv('./data/train.csv')
    
    trn_idx, val_idx = train_valid_split(df_train, 0.2)
    
    batch_size = 32
    
    x_trn = train_gen.flow_from_df(df_train.iloc[trn_idx].reset_index(),
                                   './data/train-jpg/',
                                   '.jpg',
                                   mode='fit',
                                   batch_size=batch_size)
    x_val = valid_gen.flow_from_df(df_train.iloc[val_idx].reset_index(),
                                   './data/train-jpg/',
                                   '.jpg',
                                   mode='predict',
                                   batch_size=batch_size)
    model.fit_generator(x_trn,
                        steps_per_epoch = len(trn_idx) / batch_size,
                        epochs=1,
                        workers=6, pickle_safe=True
                       )
    
    ypreds = model.predict_generator(x_val,
                                     steps = len(val_idx)/batch_size,
                                     workers=6, pickle_safe=True
                                    )
    
    mlb = train_gen.getLabelEncoder()
    predictions = ypreds > 0.2
    true_labels = mlb.transform(df_train['tags'].iloc[val_idx].values)
    
    score=fbeta_score(true_labels, predictions, beta=2, average='samples')
    
    end_global_timer = timer()
    print("################## Success #########################")
    print("Total elapsed time: %s" % (end_global_timer - global_timer))