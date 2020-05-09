import pandas as pd
import numpy as np
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

def info_df(df:pd.DataFrame) -> pd.DataFrame:
    """return information of dataframe
    
    Arguments:
        df {dataframe} -- [dataframe that you want to check info]
    
    Returns:
        pd.DataFrame -- [summary of input dataframe]
    """
    return pd.DataFrame({
        "uniques": df.nunique(),
        "nulls": df.isnull().sum(),
        "nulls (%)": df.isnull().sum() / len(df)
    }).T


def reduce_mem_usage(df:pd.DataFrame, verbose=True) -> pd.DataFrame:
    """reduce input dataframe's memory
    
    Arguments:
        df {pd.DataFrame} -- dataframe that you want to reduce mem_usage
    
    Keyword Arguments:
        verbose {bool} -- if True, print message (default: {True})
    
    Returns:
        pd.DataFrame -- dataframe that is reduced mem_usage from input dataframe
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def generate_rag_feature(df:pd.DataFrame, shift_size:int, window_size:int, min_periods=None, method='mean') -> pd.DataFrame:
    """generate rag feature
    Examples:
        All data should be long format
                                     A                      B
        0                           3.0                    2.0
        1                           0.0                    2.0
        2                           0.0                    2.0
        ...                         ...                    ...
        98                          3.0                    2.0
        99                          0.0                    2.0
        100                         0.0                    2.0


    Arguments:
        df {pd.DataFrame} -- data source of rag feature
        shift_size {int} -- size of shift 
        window_size {int} -- size of moving window
    
    Keyword Arguments:
        min_periods {int} -- Minimum number of observations in window required to have a value (otherwise result is NA).  (default: {None})
        method {str} -- method of rag_feature(mean, max, min) (default: {'mean'})
    
    Returns:
        pd.DataFrame -- [dataframe that has rag feature]
    """
    if method == 'max':
        return df.shift(shift_size).rolling(window=window_size, min_periods=min_periods).max()
    elif method == 'min':
        return df.shift(shift_size).rolling(window=window_size, min_periods=min_periods).min()
    elif method == 'mean':
        return df.shift(shift_size).rolling(window=window_size, min_periods=min_periods).mean()
    else:
        return df.shift(shift_size).rolling(window=window_size, min_periods=min_periods).mean()