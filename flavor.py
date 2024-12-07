from parallel import pmap_df
import pandas as pd
import pandas_flavor as pf
import numpy as np
import polars as pl
import janitor
# #define and register your custom functionality
# @pl.api.register_expr_namespace('custom')
# class CustomStringMethodsCollection:
#     def __init__(self, expr: pl.Expr):
#         self._expr = expr

#     def to_title_case(self) -> pl.Expr:
#         convert_to_title = (
#             pl.element().str.slice(0, 1).str.to_uppercase()
#             +
#             pl.element().str.slice(1).str.to_lowercase()
#             )

#         return (
#             self._expr.str.split(' ')
#             .arr.eval(convert_to_title)
#             .arr.join(separator=' ')
#         )

#     def replace_empty_string_with_null(self) -> pl.Expr:
#         return (self._expr.pl.when(pl.col(pl.String).str.len_chars() == 0)
#         .then(None)
#         .otherwise(pl.col(pl.String))
#         .name.keep())


# def replace_empty_string_with_null(df):
#     return df.with_columns(
#         pl.when(pl.col(pl.String).str.len_chars() == 0)
#         .then(None)
#         .otherwise(pl.col(pl.String))
#         .name.keep())

# def drop_all_null_columns(df):
#     return df[[s.name for s in df if s.null_count() != df.height]]




@pf.register_dataframe_method
def to_polars(df, **kwargs):
    return pl.from_pandas(df, **kwargs)


@pf.register_dataframe_method
def deconc(df, **kwargs):
    return pl.from_pandas(df, **kwargs)


@pf.register_dataframe_method
def str_drop_after(df, column_name: str,  pat: str, drop: bool = True):
    """Wrapper around df.str.replace"""
    split = df[column_name].str.split(pat=pat, expand=True)
    if drop:
        return df.assign(**{column_name: split[0]})
    else:
        return df.assign(**{
            f"{column_name}_left": split[0],
            f"{column_name}_right": split[1]
        })


@pf.register_dataframe_method
def str_remove(df, column_name: str, pat: str, *args, **kwargs):
    """Wrapper around df.str.replace"""
    return df.assign(**{column_name: df[column_name].str.replace(pat, "", *args, **kwargs)})


@pf.register_dataframe_method
def str_replace(df, column_name: str, pat_from: str, pat_to: str, *args, **kwargs):
    """Wrapper around df.str.replace"""
    return df.assign(**{column_name: df[column_name].str.replace(pat_from, pat_to, *args, **kwargs)})


@pf.register_dataframe_method
def str_trim(df, column_name: str, *args, **kwargs):
    """Wrapper around df.str.strip"""
    return df.assign(**{column_name: df[column_name].str.strip(*args, **kwargs)})


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
    return df.assign(**{column_name: df[column_name].str.split(pat).str[start:stop]})


@pf.register_dataframe_method
def str_join(df, column_name: str, sep: str, *args, **kwargs):
    """
    Wrapper around `df.str.join`
    Joins items in a list.
    """
    return df.assign(**{column_name: df[column_name].str.join(sep)})


@pf.register_dataframe_method
def str_split_select(
    df,
    column_name: str,
    sep: str,
    idx: int = 0,
    stop: int = None,
    autoname=None,
    drop=True
):
    """
    Wrapper around `df.str.split` with additional `idx` and `end` arguments
    to select a slice of the list of words.
    """
    name = autoname or column_name
    if stop is None:
        stop = idx + 1
    names = [f'{name}_{i}' for i in range(idx, stop)]
    split_result = df[column_name].str.split(sep, expand=True).iloc[:, idx:stop]
    new_df = df.assign(**{name: split_result[i] for i, name in enumerate(names)})
    return new_df.drop(columns=[column_name]) if drop else new_df


@pf.register_dataframe_method
def str_slice(
    df, column_name: str, start: int = None, stop: int = None, *args, **kwargs
):
    """
    Wrapper around `df.str.slice`
    """
    return df.assign(**{column_name: df[column_name].str[start:stop]})


@pf.register_dataframe_method
def highlight_best(df,
                   col,
                   criterion=np.max,
                   style='background: lightgreen'
                   ):
    # other useful styles: 'font-weight: bold'
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
    best = df.apply(criterion)[col]
    return df.style.apply(
        lambda x: [style if (x[col] == best) else '' for _ in x], axis=1
    )


@pf.register_dataframe_method
def print_full(df):
    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.precision', 3,
            'display.max_colwidth', None):
        print(df)


@pf.register_dataframe_method
def remove_boring(df):
    non_null_cols = df.dropna(axis=1, how='all').columns
    interesting_cols = [i for i in non_null_cols if len(set(df[i])) > 1]
    return df.loc[:, interesting_cols]


@pf.register_dataframe_method
@pf.register_series_method
def add_outer_index(df, value, name):
    return pd.concat({value: df}, names=[name])


@pf.register_dataframe_method
def add_outer_column(df, value):
    df.columns = pd.MultiIndex.from_arrays([[value]*len(df.columns), df.columns])
    return df


@pf.register_dataframe_method
def str_get_numbers(df, column_name: str):
    """Wrapper around df.str.extract"""
    return df.assign(**{column_name: df[column_name].str.extract(r'(\d+)', expand=False)})


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
    return (df
            .drop(column_name, 1)
            .join(pd.DataFrame(df[column_name].values.tolist(),
                               index=df.index,
                               columns=output_column_names)))


@pf.register_dataframe_method
def get_nth_element(df, column_name, n, new_column_name, drop=False):
    """
    #     AMOUNT    column_name
    #     0       [1,2,3]
    #     1       [1,2,3]
    #     2       [1,2,3]

    to (n=1)

    #     AMOUNT  column_name     new_column_name
    #     0       [1,2,3]         2
    #     1       [1,2,3]         2
    #     2       [1,2,3]         2
    """
    result = df.assign(**{new_column_name: df[column_name].str[n]})
    return result.drop(columns=[column_name]) if drop else result


@pf.register_dataframe_method
def process_dictionary_column(df, column_name):
    if column_name in df.columns:
        return (df
                .join(df[column_name].apply(pd.Series))
                .drop(column_name, 1))
    else:
        return df

@pf.register_dataframe_method
def ppipe(df, f, **kwargs):
    return pmap_df(f, df, **kwargs)



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

# from https://pyjanitor.readthedocs.io/notebooks/anime.html


@pf.register_dataframe_method
def pipeprint(df, msg, **kwargs):
    """
    print statements in a pandas pipe
    """
    print(msg)
    return df
