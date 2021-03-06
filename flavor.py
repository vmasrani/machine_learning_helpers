import pandas as pd
from parallel import pmap, pmap_df
import pandas_flavor as pf
import numpy as np

@pf.register_dataframe_method
def highlight_best(df,
                   col,
                   criterion=np.max,
                   style='background: lightgreen'
                  ):
    # other useful styles: 'font-weight: bold'
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
    best = df.apply(criterion)[col]
    return df.style.apply(lambda x: [style if (x[col] == best) else '' for i in x], axis=1)


@pf.register_dataframe_method
def remove_boring(df):
    df = df.dropna(1, how='all')
    return df[[i for i in df if len(set(df[i])) > 1]]

@pf.register_dataframe_method
@pf.register_series_method
def add_outer_index(df, value, name):
    return pd.concat({value: df}, names=[name])

@pf.register_dataframe_method
def ppipe(df, f, **kwargs):
    return pmap_df(f, df, **kwargs)

@pf.register_dataframe_method
def str_get_numbers(df, column_name: str):
    """Wrapper around df.str.replace"""

    df[column_name] = df[column_name].str.extract('(\d+)', expand=False)
    return df

@pf.register_dataframe_method
def str_drop_after(df, pat, column_name: str):
    """Wrapper around df.str.replace"""

    df[column_name] = df[column_name].str.split(pat='[', expand=True)
    return df

@pf.register_dataframe_method
def expand_list_column(df, column_name, output_column_names):
    """

    #     AMOUNT    LISTCOL
    #     0       [1,2,3]
    #     1       [1,2,3]
    #     2       [1,2,3]

    to

    #     AMOUNT  col_1     col_2    col_3
    #     0       1           2         3
    #     1       1           2         3
    #     2       1           2         3
    """

    new = pd.DataFrame(df[column_name].values.tolist(), index=df.index, columns=output_column_names)
    old = df.drop(column_name, 1)
    return old.merge(new, left_index=True, right_index=True)


# collapse_levels(sep='_')
# @pf.register_dataframe_method
# def flatten_cols(df):
#     df.columns = ['_'.join(col).strip() for col in df.columns.values]
#     return df


# to be converted to flavor when I find myself needing them

# def process_dictionary_column(df, column_name):
#     if column_name in df.columns:
#         return (df
#                 .join(df[column_name].apply(pd.Series))
#                 .drop(column_name, 1))
#     else:
#         return df


# def process_tuple_column(df, column_name, output_column_names):
#     if column_name in df.columns:
#         return df.drop(column_name, 1).assign(**pd.DataFrame(df[column_name].values.tolist(), index=df.index))
#     else:
#         return df


# def process_list_column(df, column_name, output_column_names):
#     if column_name in df.columns:
#         new = pd.DataFrame(df[column_name].values.tolist(), index=df.index, columns=output_column_names)
#         old = df.drop(column_name, 1)
#         return old.merge(new, left_index=True, right_index=True)
#     else:
#         return df


# def show_uniques(df):
#     for col in df:
#         print(f'{col}: ', df[col].unique())


# def highlight_best(df, col):
#     best = df[col].max()
#     return df.style.apply(lambda x: ['background: lightgreen' if (x[col] == best) else '' for i in x], axis=1)


# def filter_uninteresting(df):
#     df = df.dropna(1, how='all')
#     return df[[i for i in df if len(set(df[i])) > 1]]

def pgroupby(df, groups, f,  **kwargs):
    '''# mirror groupby order (group then agg)
    replace:
        results = df.groupby(['col1','col2']).apply(f)
    with:
        results = df.pgroupby(['col1','col2'], f)
    '''
    # split into names and groups
    names, df_split = zip(*[(n,g) for n,g in df.groupby(groups)])
    # pmap groups
    out = pmap(f, df_split, **kwargs)
    # reassemble and return
    groups = [groups] if isinstance(groups, str) else groups
    return pd.concat([pd.concat({k: v}, names=groups) for k, v in zip(names, out)])


# from https://pyjanitor.readthedocs.io/notebooks/anime.html

@pf.register_dataframe_method
def str_remove(df, column_name: str, pat: str, *args, **kwargs):
    """Wrapper around df.str.replace"""

    df[column_name] = df[column_name].str.replace(pat, "", *args, **kwargs)
    return df


@pf.register_dataframe_method
def str_replace(df, column_name: str, pat_from: str, pat_to: str,  *args, **kwargs):
    """Wrapper around df.str.replace"""

    df[column_name] = df[column_name].str.replace(pat_from, pat_to, *args, **kwargs)
    return df



@pf.register_dataframe_method
def str_trim(df, column_name: str, *args, **kwargs):
    """Wrapper around df.str.strip"""

    df[column_name] = df[column_name].str.strip(*args, **kwargs)
    return df


@pf.register_dataframe_method
def str_word(
    df,
    column_name: str,
    start: int = None,
    stop: int = None,
    pat: str = " ",
    *args,
    **kwargs
):
    """
    Wrapper around `df.str.split` with additional `start` and `end` arguments
    to select a slice of the list of words.
    """

    df[column_name] = df[column_name].str.split(pat).str[start:stop]
    return df


@pf.register_dataframe_method
def str_join(df, column_name: str, sep: str, *args, **kwargs):
    """
    Wrapper around `df.str.join`
    Joins items in a list.
    """

    df[column_name] = df[column_name].str.join(sep)
    return df


@pf.register_dataframe_method
def str_slice(
    df, column_name: str, start: int = None, stop: int = None, *args, **kwargs
):
    """
    Wrapper around `df.str.slice
    """

    df[column_name] = df[column_name].str[start:stop]
    return df
