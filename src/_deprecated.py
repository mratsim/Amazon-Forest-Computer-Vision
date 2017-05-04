### From Validation.py

## DEPRECATED: Unfortunately COBYLA from Scipy can does not respect "lexical bounds".
## Beware: the following will probably overfit the threshold to the validation set
##################################################################################
## Metrics
## Given the labels imbalance we can't use the same threshold for each label.
## We could implement our own maximizer on all 17 classes but scipy.optimize already have
## 4 optimizations algorithms in C/Fortran that can work with constraints: L-BFGS-B, TNC, COBYLA and SLSQP.
## Of those only cobyla doesn't rely on 2nd order hessians which are error-prone with our function
## based on inequalities

# Cobyla constraints are build by comparing return value with 0.
# They must be >= 0 or be rejected

def constr_sup0(x):
    return np.min(x)
def constr_inf1(x):
    return 1 - np.max(x)

def f2_score(true_target, predictions):

    def f_neg(threshold):
        ## Scipy tries to minimize the function so we must get its inverse
        return - fbeta_score(true_target, predictions > threshold, beta=2, average='samples')

    # Initialization of best threshold search
    thr_0 = np.array([0.2 for i in range(17)])
    
    # Search
    thr_opt = fmin_cobyla(f_neg, thr_0, [constr_sup0,constr_inf1], disp=0)

    logger.info("===> Optimal threshold for each label:\n{}".format(thr_opt))
    
    score = fbeta_score(true_target, predictions > thr_opt, beta=2, average='samples')
    return score, thr_opt

## The jit is slower than scikit by a few ms. Unless the optimizing loop can be JIT too it's not worth it

##################################################################################
## Metrics
## Given the labels imbalance we can't use the same threshold for each label.
## We loop on each column label independently and maximize F2 score
## Limit: might overfit
## We don't model interdependance of coefs

from numba import jit


# True Positive
@jit(nopython=True)
def true_pos(pred_labels, true_labels):
    return np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
 
# True Negative
@jit(nopython=True)
def true_neg(pred_labels, true_labels):
    return np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
 
# False Positive - Type I Error
@jit(nopython=True)
def false_pos(pred_labels, true_labels):
    return np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
 
# False Negative - Type II Error
@jit(nopython=True)
def false_neg(pred_labels, true_labels):
    return np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

@jit(nopython=True)
def precision(pred_labels, true_labels):
    TP = true_pos(pred_labels, true_labels)
    FP = false_pos(pred_labels, true_labels)
    
    # Edge cases True Positives = 0, False negative = 0
    # No predicted labels at all
    # Shouldn't happen all photos must have at least one label
    # We return 0 so that the threshold becomes better
    # Should we penalize more ?
    if TP==0 and FP==0: return 0
    
    return TP / (TP + FP)

@jit(nopython=True)
def recall(pred_labels, true_labels):
    TP = true_pos(pred_labels, true_labels)
    FN = false_neg(pred_labels, true_labels)
    
    # Edge cases True Positives = 0, False negative = 0
    # i.e no label in the true_labels input.
    # Shouldn't happen  all photos have at least one label

    return TP / (TP + FN)

@jit(nopython=True)
def f2_score_macro(pred_labels, true_labels):
    p = precision(pred_labels, true_labels)
    r = recall(pred_labels, true_labels)
    if p == 0 and r == 0: return 0
    return 5 * p * r / (4 * p + r)

@jit
def f2_score_mean(pred_labels, true_labels):
    # F2_score_mean accelerated by numba
    # Cannot force nopython mode because for loop on arrays does not work
    i = 0
    acc = 0
    for (x,y) in zip(pred_labels,true_labels):
        acc += f2_score_macro(x,y)
        i+=1
    return acc / i


### Kaggle kernel search
def search_best_threshold(p_valid, y_valid, try_all=False, verbose=False):
    p_valid, y_valid = np.array(p_valid), np.array(y_valid)

    best_threshold = 0
    best_score = -1
    totry = np.arange(0,1,0.005) if try_all is False else np.unique(p_valid)
    for t in totry:
        score = fbeta_score(y_valid, p_valid > t, beta=2, average='samples')
        if score > best_score:
            best_score = score
            best_threshold = t
    logger.info("===> Optimal threshold for each label:\n{}".format(best_threshold))
    return best_score, best_threshold

# Search with L-BFGS-B
    thr_0 = np.array([0.20 for i in range(17)])
    constraints = [(0.,1.) for i in range(17)]
thr_opt, score_neg, dico = fmin_l_bfgs_b(f_neg, thr_0, bounds=constraints, approx_grad=True, epsilon=0.05)

## From dataload.py
##################################################
## DEPRECATED: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/8?u=mratsim
## Augmentation on PyTorch are done randomly at each epoch

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

        self.X = tmp_df['image_name']
        self.y = self.mlb.fit_transform(tmp_df['tags'].str.split()).astype(np.float32)
        self.augmentNumber = 14 # TODO, do something about this harcoded value

    def __getitem__(self, index):
        real_length = self.real_length()
        real_index = index % real_length
        
        img = Image.open(self.img_path + self.X[real_index] + self.img_ext)
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
        
        label = from_numpy(self.y[real_index])
        return img, label
    
    def __len__(self):
        return len(self.X.index) * self.augmentNumber
    
    def real_length(self):
        return len(self.X.index)
    
    def getLabelEncoder(self):
        return self.mlb
    
#### Usage

    ############################################################
    # Augmented part
    # X_train = AugmentedAmazonDataset('./data/train.csv','./data/train-jpg/','.jpg',
    #                            ds_transform
    #                            )
    
    # Creating a validation split
    # train_idx, valid_idx = augmented_train_valid_split(X_train, 0.2)
    
    # nb_augment = X_train.augmentNumber
    # augmented_train_idx = [i * nb_augment + idx for idx in train_idx for i in range(0,nb_augment)]
                           
    # train_sampler = SubsetRandomSampler(augmented_train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)
    ###########################################################
    
    
##################################################
## DEPRECATED: AugmentedAmazonDataset is deprecated
## https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/8?u=mratsim
## Augmentation on PyTorch are done randomly at each epoch


def augmented_train_valid_split(dataset, test_size = 0.25, shuffle = False, random_seed = 0):
    """ Return a list of splitted indices from a DataSet.
    Indices can be used with DataLoader to build a train and validation set.
    
    Arguments:
        A Dataset
        A test_size, as a float between 0 and 1 (percentage split) or as an int (fixed number split)
        Shuffling True or False
        Random seed
    """
    length = dataset.real_length()
    indices = list(range(1,length))
    
    if shuffle == True:
        random.seed(random_seed)
        random.shuffle(indices)
    
    if type(test_size) is float:
        split = floor(test_size * length)
    elif type(test_size) is int:
        split = test_size
    else:
        raise ValueError('%s should be an int or a float' % str)
    return indices[split:], indices[:split]
