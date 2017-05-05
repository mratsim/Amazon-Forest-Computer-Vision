from torch.utils.data.sampler import Sampler
import numpy as np
import pandas as pd

class SubsetSampler(Sampler):
     """Samples elements from a given list of indices.
 
     Arguments:
         indices (list): a list of indices
     """
 
     def __init__(self, indices):
        self.num_samples = len(indices)
        self.indices = indices
 
     def __iter__(self):
        return iter(self.indices)
 
     def __len__(self):
        return self.num_samples
    
def balance_weights(df_source, col_target, mlb):
    """ Compute balanced weights from a Multilabel dataframe
    
    Arguments:
        Dataframe
        The name of the column with the target labels
        A MultiLabelBinarizer to one-hot-encode/decode the label column
        
    Returns:
        A Pandas Series with balanced weights
    """
    
    # Create a working copy of the dataframe
    df = df_source.copy(deep=True)
    
    df_labels = mlb.transform(df[col_target].str.split(" "))
    
    ## Next 4 lines won't be needed when axis argument is added to np.unique in NumPy 1.13
    ncols = df_labels.shape[1]
    dtype = df_labels.dtype.descr * ncols
    struct = df_labels.view(dtype)
    uniq_labels, uniq_counts = np.unique(struct, return_counts=True)
    
    uniq_labels = uniq_labels.view(df_labels.dtype).reshape(-1, ncols)
    
    ## We convert the One-Hot-Encoded labels as string to store them in a dataframe and join on them
    df_stats = pd.DataFrame({
        'target':np.apply_along_axis(np.array_str, 1, uniq_labels),
        'freq':uniq_counts
    })
    
    df['target'] = np.apply_along_axis(np.array_str, 1, df_labels)
    
    ## Join the dataframe to add frequency
    df = df.merge(df_stats,how='left',on='target')
    
    ## Compute balanced weights
    weights = 1 / df['freq'].astype(np.float)
    
    return weights