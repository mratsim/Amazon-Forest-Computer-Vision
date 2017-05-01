from __future__ import absolute_import
from __future__ import print_function

from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.image import ImageDataGenerator, Iterator, load_img, img_to_array
import pandas as pd
import os
import threading
import numpy as np
import keras.backend as K

## For computing mean and std
from tqdm import tqdm
import cv2

class AmazonGenerator(ImageDataGenerator):
    def __init__(self, *args, **kwargs):
        super(AmazonGenerator, self).__init__(*args, **kwargs)
        self.iterator = None
    
    def flow_from_csv(self, csv_path, img_path, img_ext,
                     mode='fit',
                     target_size=(256, 256),
                     color_mode='rgb',
                     batch_size=32, shuffle=True, seed=None):
 
        self.iterator = AmazonCSVIterator(self, csv_path,
                              img_path, img_ext,
                              mode=mode,
                              target_size = target_size,
                              color_mode = color_mode,
                              batch_size = batch_size,
                              shuffle = shuffle,
                              seed = seed,
                              data_format=None)
        self.mlb = self.iterator.getLabelEncoder()
        return(self.iterator)
    
    def flow_from_df(self, dataframe, img_path, img_ext,
                     mode='fit',
                     target_size=(256, 256),
                     color_mode='rgb',
                     batch_size=32, shuffle=True, seed=None):
 
        self.iterator = AmazonDFIterator(self, dataframe,img_path, img_ext,
                              mode=mode,
                              target_size = target_size,
                              color_mode = color_mode,
                              batch_size = batch_size,
                              shuffle = shuffle,
                              seed = seed,
                              data_format=None)
        self.mlb = self.iterator.getLabelEncoder()
        return(self.iterator) 
    
    def getLabelEncoder(self):
        return self.iterator.getLabelEncoder()
    
    def fit_from_csv(self, csv_path, img_path, img_ext, rescale, target_size):
        '''Required for featurewise_center, featurewise_std_normalization 
        when using images loaded from csv.

        # Arguments
            csv_path: Path to the csv with image list
            img_path: Directory with all images
            img_ext: Extension of images
            rescaling factor: usually we rescale images from 0-255 to 0-1
            resolution: A tuple of int. Images will be rescaled to that resolution before computing mean as we need to hold them all in memory. Set as big as your memory allows
        '''
        
        # Computing mean and variance using Welford's algorithm for one pass only and numerical stability.        
        df = pd.read_csv(csv_path)
        
        # Pre-allocation
        shape = cv2.imread(os.path.join(
                             img_path,
                             df['image_name'].iloc[0] + img_ext)).shape
        
        mean= np.zeros(shape, dtype=np.float32)
        M2= np.zeros(shape, dtype=np.float32)

        print('Computing mean and standard deviation on the dataset')
        for n, file in enumerate(tqdm(df['image_name'], miniters=256), 1):
            img = cv2.imread(os.path.join(img_path, file + img_ext)).astype(np.float32)
            img *= rescale
            delta = img - mean
            mean += delta/n
            delta2 = img - mean
            M2 += delta*delta2
                
        self.mean = mean
        self.std = M2 / (n-1)
        
        print("Mean has shape: " + str(self.mean.shape))
        print("Std has shape: " + str(self.std.shape))
        
    def dump_dataset_mean_std(self, path_mean, path_std):
        if self.mean is None or self.std is None:
            raise ValueError('Mean and Std must be computed before, fit the generator first')
        np.save(path_mean, self.mean)
        np.save(path_std, self.std)
        

    def load_mean_std(self, path_mean, path_std):
        self.mean = np.load(path_mean)
        self.std = np.load(path_std)
        print("Mean has shape: " + str(self.mean.shape))
        print("Std has shape: " + str(self.std.shape))

class AmazonCSVIterator(Iterator):
    def __init__(self, image_data_generator, csv_path,
                 img_path, img_ext,
                 mode='fit',
                 target_size=(256, 256),
                 color_mode='rgb',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None):
        
        ## Common initialization routines
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        
        if data_format is None:
            self.data_format = K.image_data_format()

        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
                
        self.image_data_generator = image_data_generator
        
        ## Specific to Amazon
        tmp_df = pd.read_csv(csv_path)
        assert tmp_df['image_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
"Some images referenced in the CSV file were not found"
        
        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.img_ext = img_ext
        self.X = tmp_df['image_name']
        self.mode = mode
        if mode == 'fit':
            self.y = self.mlb.fit_transform(tmp_df['tags'].str.split())
        
        ## Init parent class
        super(AmazonCSVIterator, self).__init__(self.X.shape[0],
                                             batch_size, shuffle, seed)

    def next(self):
        """For python 2.x.
        # Returns The next batch.
        """
        
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        
        # Build batch of images
        for i, j in enumerate(index_array):
            fpath = os.path.join(self.img_path,self.X[j] + self.img_ext)
            img = load_img(fpath,
                           grayscale=grayscale,
                           target_size=self.target_size)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        
        # Build batch of labels.
        if mode=='fit':
            batch_y = self.y[index_array]
            return batch_x, batch_y
        elif mode=='predict':
            return batch_x
        else: raise ValueError('The mode should be either \'fit\' or \'predict\'')
            
    def getLabelEncoder(self):
        return self.mlb

class AmazonDFIterator(Iterator):
    def __init__(self, image_data_generator, df, img_path, img_ext,
                 mode='fit',
                 target_size=(256, 256),
                 color_mode='rgb',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None):
        
        ## Common initialization routines
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        
        if data_format is None:
            self.data_format = K.image_data_format()

        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
                
        self.image_data_generator = image_data_generator
        
        ## Specific to Amazon
        assert df['image_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
"Some images referenced in the CSV file were not found"
        
        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.img_ext = img_ext
        self.X = df['image_name']
        self.mode = mode
        if mode == 'fit':
            self.y = self.mlb.fit_transform(df['tags'].str.split())
        
        ## Init parent class
        super(AmazonDFIterator, self).__init__(self.X.shape[0],
                                             batch_size, shuffle, seed)

    def next(self):
        """For python 2.x.
        # Returns The next batch.
        """
        
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        
        # Build batch of images
        for i, j in enumerate(index_array):
            fpath = os.path.join(self.img_path,self.X[j] + self.img_ext)
            img = load_img(fpath,
                           grayscale=grayscale,
                           target_size=self.target_size)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        
        # Build batch of labels.
        if self.mode=='fit':
            batch_y = self.y[index_array]
            return batch_x, batch_y
        elif self.mode=='predict':
            return batch_x
        else: raise ValueError('The mode should be either \'fit\' or \'predict\'')
            
    def getLabelEncoder(self):
        return self.mlb