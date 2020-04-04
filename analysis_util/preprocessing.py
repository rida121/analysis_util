import pandas as pd
from typing import Optional
import umap

def target_encoding(train_df:pd.DataFrame, test_df:pd.DataFrame, target_key:str, encoding_keys:list, method='mean') -> pd.DataFrame:
    """do target encoding. encoded column name is enc_ + method + '_' + encoding_key
    
    Arguments:
        train_df {pd.DataFrame} -- [df for training]
        test_df {pd.DataFrame} -- [df for test]
        target_key {str} -- [object key name]
        encoding_keys {list} -- [list of key names that you want to encode]
    
    Keyword Arguments:
        method {str} -- [how to encode 'mean', median', 'mode'] (default: {'mean'})
    
    Returns:
        pd.DataFrame -- [encoded test dataframe]
    """

    for encoding_key in encoding_keys:
        enc_key_name = 'enc_' + method + '_' + encoding_key
        if method is 'mean':
            encoding_value = train_df.groupby(encoding_key)[target_key].mean()
        elif method is 'median':
            encoding_value = train_df.groupby(encoding_key)[target_key].median()
        else:
            encoding_value = train_df.groupby(encoding_key)[target_key].mean()
        test_df.loc[:, enc_key_name] = test_df.loc[:, encoding_key].map(encoding_value)
    # move target_key to last
    column_list = test_df.columns.to_list()
    column_list.remove(target_key)
    column_list.append(target_key)
    return test_df[column_list]

def dimensionality_reduction(df:pd.DataFrame, method='umap', n_neighbors=100) -> pd.DataFrame:
    """add umap features to input df
    
    Arguments:
        df {pd.DataFrame} -- [input df]
    
    Keyword Arguments:
        method {str} -- [what dim reduction method you use] (default: {'umap'})
    
    Returns:
        pd.DataFrame -- [input df + umap features]
    """
    um = umap.UMAP(n_neighbors=n_neighbors, n_components=2)
    um.fit(df)
    tmp = um.transform(df.values)
    um_df = pd.DataFrame(tmp, columns=['dim_x', 'dim_y'])
    df.reset_index(drop=True, inplace=True)
    return pd.concat([df, um_df], axis=1)


