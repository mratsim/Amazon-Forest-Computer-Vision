import xgboost as xgb
import cv2
import mxnet as mx
import os
import numpy as np
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split

SRC_IMAGES = '../data/train-jpg/'
SRCDIR = os.listdir(SRC_IMAGES)
TMPDIR = './tmp/'

def get_extractor():
    model = mx.model.FeedForward.load('../pretrained-models/resnet-50', 0, ctx=mx.gpu(), numpy_batch_size=1)
    fea_symbol = model.symbol.get_internals()["flatten0_output"]
    feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, numpy_batch_size=64,
                                             arg_params=model.arg_params, aux_params=model.aux_params,
                                             allow_extra_params=True)

    return feature_extractor


def prepare_image_batch(image):
    img = SRC_IMAGES + image
    img = cv2.imread(img)
    img = 255.0 / np.amax(img) * img
    # img = cv2.equalizeHist(img.astype(np.uint8))
    img = cv2.resize(img.astype(np.int16), (224, 224))
    img = img.reshape(1,3,224,224)

    return img

def calc_features():
    net = get_extractor()
    n=1
    for image in SRCDIR:
        print("Doing image %s/%s: %s" % (n, len(SRCDIR), image))
        img = prepare_image_batch(image)
        print(img.shape)
        feats = net.predict(img)
        print("Prediction features have shape:")
        print(feats.shape)
        np.save(TMPDIR+image, feats)
        
        n+=1


if __name__ == '__main__':
    start_time = timer()
    calc_features()
    # make_submit()
    end_time = timer()
    print("Elapsed time: %s" % (end_time - start_time))