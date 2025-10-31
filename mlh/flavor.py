import ast
import pandas as pd
from sqlalchemy import inspect
import contextlib
from rapidfuzz import fuzz
from filecmp import cmp
import json
from .parallel import pmap_df
import pandas_flavor as pf
import numpy as np
import polars as pl
import janitor
from .ml_helpers import parse_str_to_json

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


# # Function to convert string to appropriate data structure
# def _convert_to_structure(value):
#     # If the value is already a list, dict, int, or float, return it as is
#     if isinstance(value, (list, dict, int, float)):
#         return value

#     # If the value is a string, attempt to parse it
#     if isinstance(value, str):
#         try:
#             # Attempt to evaluate the string as a Python literal
#             return ast.literal_eval(value)
#         except (ValueError, SyntaxError):
#             # Handle malformed strings by attempting a more lenient parsing
#             if '[' in value:
#                 formatted_string = value.replace('[', '["').replace(']', '"]').replace(', ', '", "')
#                 return ast.literal_eval(formatted_string)
#     # Return the value as is if it doesn't match any of the above conditions
#     return value

@pf.register_dataframe_method
def append_df_aligned(
    df: pd.DataFrame,
    table: str,
    engine,
    *,
    chunksize: int = 10_000,
    method: str = "multi",
) -> int:
    """
    Append a DataFrame to an existing PostgreSQL table, aligning columns:
      - Adds NA for any table columns missing in df
      - Keeps all df columns (extra columns are ignored)
      - Reorders to match table column order for insertion
      - Creates table if it doesn't exist

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    table : str
        Target table. Accepts "schema.table" or "table".
    engine : sqlalchemy.Engine | sqlalchemy.Connection
        SQLAlchemy engine/connection to the target database.
    chunksize : int
        Rows per batch for to_sql.
    method : str
        to_sql insert method. "multi" groups many rows per INSERT.

    Returns
    -------
    int
        Number of rows appended.
    """
    # Parse schema.table
    if "." in table and not table.strip().startswith('"'):
        schema, tbl = table.split(".", 1)
    else:
        schema, tbl = None, table

    insp = inspect(engine)

    # Check if table exists
    table_exists = insp.has_table(tbl, schema=schema)

    if not table_exists:
        # Table doesn't exist, create it with the DataFrame structure
        df = df.copy()
        df.columns = [str(c) for c in df.columns]

        df.to_sql(
            name=tbl,
            con=engine,
            schema=schema,
            if_exists="replace",
            index=False,
            chunksize=chunksize,
            method=method,
        )
    else:
        # Table exists, align columns and append
        cols_meta = insp.get_columns(tbl, schema=schema)
        target_cols = [c["name"] for c in cols_meta]

        df = df.copy()
        df.columns = [str(c) for c in df.columns]

        # Add NA for any missing table columns
        for c in target_cols:
            if c not in df.columns:
                df[c] = pd.NA

        # Reorder to match table columns
        df_aligned = df[target_cols]

        # Append to existing table
        df_aligned.to_sql(
            name=tbl,
            con=engine,
            schema=schema,
            if_exists="append",
            index=False,
            chunksize=chunksize,
            method=method,
        )

    return len(df)



@pf.register_dataframe_method
def fuzzy_filter(df: pd.DataFrame, column_name: str, query: str, threshold: int) -> pd.DataFrame:
    df['score'] = df[column_name].apply(lambda x: fuzz.token_set_ratio(query, x))
    return df[df['score'] >= threshold].sort_values('score', ascending=False)


@pf.register_dataframe_method
def convert_str_to_json(df, cols=None):
    df = df.copy()
    if cols is None:
        cols = df.columns.tolist()
    if isinstance(cols, str):
        cols = [cols]

    def _safe_parse_json(col):
        with contextlib.suppress(Exception):
            return col.apply(parse_str_to_json)
        return col

    return df.assign(**{c: _safe_parse_json(df[c]) for c in cols})


@pf.register_dataframe_method
def unwrap_dict_in_list(df: pd.DataFrame, cols: list[str] | str | None = None) -> pd.DataFrame:
    def unwrap_dict(x): # type: ignore
        if isinstance(x, list) and len(x) == 1 and isinstance(x[0], dict):
            return x[0]
        return x

    return (df.transform_columns(cols or list(df.columns), unwrap_dict))


@pf.register_dataframe_method
def json_normalize_dataframe(df):
    json_str = df.unwrap_dict_in_list().to_json(orient='records')
    return pd.json_normalize(json.loads(json_str))

@pf.register_dataframe_method
def yank(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.set_index(col).reset_index()


# @pf.register_dataframe_method
# def convert_to_structure(df, cols=None):
#     if cols is None:
#         cols = df.columns.tolist()
#     if isinstance(cols, str):
#         cols = [cols]
#     return df.transform_columns(cols, _convert_to_structure)



# @pf.register_dataframe_method
# def convert_to_structure(df, cols=None):
#     if cols is None:
#         cols = df.columns.tolist()
#     if isinstance(cols, str):
#         cols = [cols]
#     return df.transform_columns(cols, _convert_to_structure)


@pf.register_dataframe_method
def ppipe(df, f, **kwargs) -> pd.DataFrame:
    return pmap_df(f, df, **kwargs)


@pf.register_dataframe_method
def to_polars(df, **kwargs) -> pl.DataFrame:
    return pl.from_pandas(df, **kwargs)


@pf.register_dataframe_method
def deconc(df, **kwargs) -> pd.DataFrame:
    return pl.from_pandas(df, **kwargs)


@pf.register_dataframe_method
def sort_columns_naturally(df) -> pd.DataFrame:
    cols = (pd
        .DataFrame(df.columns.tolist(), columns=['headers'])
        .sort_naturally('headers')
        .headers
        .tolist()
    )
    return df[cols]

@pf.register_dataframe_method
def str_drop_after(df: pd.DataFrame, column_name: str,  pat: str, drop: bool = True, drop_before=False) -> pd.DataFrame:
    """Wrapper around df.str.replace"""
    split = df[column_name].str.split(pat=pat, expand=True)
    if not drop:
        return df.assign(**{
            f"{column_name}_left": split[0],
            f"{column_name}_right": split[1]
        })
    idx = 1 if drop_before else 0
    return df.assign(**{column_name: split[idx]})


@pf.register_dataframe_method
def str_remove(df: pd.DataFrame, column_name: str, pat: str, *args, **kwargs) -> pd.DataFrame:
    """Wrapper around df.str.replace"""
    return df.assign(**{column_name: df[column_name].str.replace(pat, "", *args, **kwargs)})


@pf.register_dataframe_method
def str_replace(df: pd.DataFrame, column_name: str, pat_from: str, pat_to: str, *args, **kwargs) -> pd.DataFrame:
    """Wrapper around df.str.replace"""
    return df.assign(**{column_name: df[column_name].str.replace(pat_from, pat_to, *args, **kwargs)})


@pf.register_dataframe_method
def str_trim(df: pd.DataFrame, column_name: str, *args, **kwargs) -> pd.DataFrame:
    """Wrapper around df.str.strip"""
    return df.assign(**{column_name: df[column_name].str.strip(*args, **kwargs)})


# @pf.register_dataframe_method
# def str_word(
#     df: pd.DataFrame,
#     column_name: str,
#     start: int = None,
#     stop: int = None,
#     pat: str = " ",
#     *args,
#     **kwargs
# ):
#     """
#     Wrapper around `df.str.split` with additional `start` and `end` arguments
#     to select a slice of the list of words.
#     """
#     return df.assign(**{column_name: df[column_name].str.split(pat).str[start:stop]})


@pf.register_dataframe_method
def str_join(df: pd.DataFrame, column_name: str, sep: str, *args, **kwargs) -> pd.DataFrame:
    """
    Wrapper around `df.str.join`
    Joins items in a list.
    """
    return df.assign(**{column_name: df[column_name].str.join(sep)})


@pf.register_dataframe_method
def str_split_select(
    df: pd.DataFrame,
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
    new_df = df.assign(**{name: split_result.iloc[:, i] for i, name in enumerate(names)})
    return new_df.drop(columns=[column_name]) if drop else new_df


@pf.register_dataframe_method
def str_slice(
    df: pd.DataFrame, column_name: str, start: int = None, stop: int = None, *args, **kwargs
) -> pd.DataFrame:
    """
    Wrapper around `df.str.slice`
    """
    return df.assign(**{column_name: df[column_name].str[start:stop]})


@pf.register_dataframe_method
def highlight_best(df: pd.DataFrame,
                   col: str,
                   criterion=np.max,
                   style='background: lightgreen'
                   ) -> pd.DataFrame:
    # other useful styles: 'font-weight: bold'
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
    best = df.apply(criterion)[col]
    return df.style.apply(
        lambda x: [style if (x[col] == best) else '' for _ in x], axis=1
    )


@pf.register_dataframe_method
def print_full(df: pd.DataFrame) -> None:
    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.precision', 3,
            'display.max_colwidth', None):
        print(df)


@pf.register_dataframe_method
def remove_boring(df: pd.DataFrame) -> pd.DataFrame:
    non_null_cols = df.dropna(axis=1, how='all').columns
    interesting_cols = [i for i in non_null_cols if len(set(df[i])) > 1]
    return df.loc[:, interesting_cols]


@pf.register_dataframe_method
@pf.register_series_method
def add_outer_index(df: pd.DataFrame, value: str, name: str) -> pd.DataFrame:
    return pd.concat({value: df}, names=[name])


@pf.register_dataframe_method
def add_outer_column(df: pd.DataFrame, value: str) -> pd.DataFrame:
    df.columns = pd.MultiIndex.from_arrays([[value]*len(df.columns), df.columns])
    return df


@pf.register_dataframe_method
def str_get_numbers(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Wrapper around df.str.extract"""
    return df.assign(**{column_name: df[column_name].str.extract(r'(\d+)', expand=False)})


@pf.register_dataframe_method
def expand_list_column(df: pd.DataFrame, column_name: str, output_column_names: list[str]) -> pd.DataFrame:
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
def get_nth_element(df: pd.DataFrame, column_name: str, n: int, new_column_name: str, drop: bool = False) -> pd.DataFrame:
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
def process_dictionary_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    if column_name in df.columns:
        return (df
                .join(df[column_name].apply(pd.Series))
                .drop(column_name, 1))
    else:
        return df



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



# # Function to convert string to appropriate data structure
# def convert_to_structure(value):
#     # If the value is already a list, dict, int, or float, return it as is
#     if isinstance(value, (list, dict, int, float)):
#         return value

#     # If the value is a string, attempt to parse it
#     if isinstance(value, str):
#         try:
#             # Attempt to evaluate the string as a Python literal
#             return ast.literal_eval(value)
#         except (ValueError, SyntaxError):
#             # Handle malformed strings by attempting a more lenient parsing
#             if '[' in value:
#                 formatted_string = value.replace('[', '["').replace(']', '"]').replace(', ', '", "')
#                 return ast.literal_eval(formatted_string)

#     # Return the value as is if it doesn't match any of the above conditions
#     return value
