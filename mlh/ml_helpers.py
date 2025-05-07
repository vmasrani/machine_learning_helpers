from __future__ import division, print_function
import time
from random import uniform
from functools import wraps

from ast import literal_eval
import colorsys
import contextlib
import json
import os
import random
import socket
import ssl
import subprocess
import sys

import warnings
from collections import defaultdict
from datetime import datetime
from functools import singledispatch
from pathlib import Path
from pprint import pprint
from types import SimpleNamespace
from typing import Any, Dict, List

import janitor
import joblib
import matplotlib.colors as mc
import numpy as np
import pandas as pd
import requests

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

persist_dir = Path('./.persistdir')

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]



def retry_with_backoff(max_retries=3, initial_delay=1):
    """
    Decorator that implements exponential backoff for rate limited API calls.

    Args:
        max_retries (int): Maximum number of retries before giving up
        initial_delay (float): Initial delay in seconds before first retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            while retry_count < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retry_count += 1
                    if "RATE_LIMIT_EXCEEDED" in str(e) and retry_count < max_retries:
                        # Calculate exponential backoff delay with jitter
                        delay = initial_delay * (2 ** (retry_count - 1))
                        jitter = uniform(0, 0.1 * delay)
                        time.sleep(delay + jitter)
                    else:
                        if retry_count == max_retries:
                            print(f"Max retries ({max_retries}) reached. Last error: {e}")
                            raise
                        print(f"Error in {func.__name__}: {e}")
                        raise
            return None
        return wrapper
    return decorator


def parse_str_to_json(data):
    if isinstance(data, list):
        return [parse_str_to_json(item) for item in data]
    data = data.replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false")
    return json.loads(data)

def merge_dicts(dicts_list: List[Dict]) -> Dict:
    merged = defaultdict(list)
    remove_nones = lambda x: [i for i in x if i is not None]

    def tidy(x):
        val = list(set(flatten(remove_nones(x))))
        if len(val) == 1:
            return val[0]
        elif not val:
            return None
        else:
            return val

    for t in dicts_list:
        for k, v in t.items():
            merged[k].append(v)

    return {k: tidy(v) for k, v in merged.items()}


def nested_dict():
    return defaultdict(nested_dict)


def encode(df, cols):
    """
    Encode the values in the specified columns of a DataFrame.

    This function encodes the values in the specified columns of a DataFrame. It first creates a set of all unique values in the specified columns, then maps each unique value to an integer. It then replaces each value in the specified columns with its corresponding integer.

    Args:
        df (pandas.DataFrame): The DataFrame to encode.
        cols (list): A list of column names to encode.

    Returns:
        pandas.DataFrame: The encoded DataFrame.
        dict: A dictionary mapping integers to the original string values.
        dict: A dictionary mapping string values to integers.
    """
    nodes = {node for col in cols for node in df[col].to_list()}
    int_to_string = dict(enumerate(nodes))
    string_to_int = {v: k for k, v in int_to_string.items()}
    df_encoded = df.transform_columns(cols, lambda x: x.map(string_to_int), elementwise=False)
    return df_encoded, int_to_string, string_to_int


def read_sql_query_tqdm(query, con, chunksize=1000, **kwargs):
    """
    Read a SQL query with a progress bar.

    This function reads a SQL query with a progress bar. It first counts the total number of rows in the result set, then reads the query in chunks and updates the progress bar after each chunk.

    Args:
        query (str): The SQL query to read.
        con (sqlalchemy.engine.base.Connection): The database connection.
        chunksize (int): The number of rows to read at a time.
        **kwargs: Additional keyword arguments to pass to pd.read_sql_query.

    Returns:
        pandas.DataFrame: The result set.
    """
    count_rows = f"""
    SELECT
        COUNT(*) AS row_count
    FROM ({query}) AS subquery
    """
    # Get the total number of rows in the result set
    total_rows: Any = pd.read_sql_query(count_rows, con=con).iloc[0, 0]

    progress_bar = tqdm(total=total_rows, unit='row')

    df = pd.read_sql_query(query, con=con, chunksize=chunksize, **kwargs)

    def _update_tqdm(chunk):
        progress_bar.update(len(chunk))
        return chunk

    res = pd.concat([_update_tqdm(chunk) for chunk in df])
    progress_bar.close()
    return res



def flatten(container):
    """https://stackoverflow.com/questions/10823877/what-is-the-fastest-way-to-flatten-arbitrarily-nested-lists-in-python"""
    for i in container:
        if isinstance(i, (list, tuple)):
            yield from flatten(i)
        else:
            yield i


def scale(x, out_range=(-1, 1)):
    """ https://codereview.stackexchange.com/questions/185785/scale-numpy-array-to-certain-range"""
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def hits_and_misses(y_hat, y_testing):
    tp = sum(y_hat + y_testing > 1)
    tn = sum(y_hat + y_testing == 0)
    fp = sum(y_hat - y_testing > 0)
    fn = sum(y_testing - y_hat > 0)
    return tp, tn, fp, fn


def get_auc(roc):
    prec = roc['prec'].fillna(1)
    recall = roc['recall']
    return metrics.auc(recall, prec)


def classification_metrics(labels, preds):
    tp, tn, fp, fn = hits_and_misses(labels, preds)

    precision = tp / (tp + fp) if (tp + fp) != 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) != 0 else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) != 0 else np.nan

    f1 = 2.0 * (precision * recall / (precision + recall)) if (precision + recall) != 0 else np.nan

    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "prec": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        'acc': accuracy_score(preds, labels),
    }


def block_print():
    # pylint: disable=R1732,W1514
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__



def train_test_val(data, splits=(0.7, 0.2, 0.1)):
    train_p, test_p, val_p = splits
    train, testval = train_test_split(data, train_size=train_p)
    if val_p == 0:
        return train, testval
    else:
        test, val = train_test_split(testval, train_size=test_p / (test_p + val_p))
    return train, test, val


def group_train_test_val(data: pd.DataFrame, group: str, **kwargs):
    groups = data[group]
    return [data[groups.isin(split)] for split in train_test_val(groups.unique(), **kwargs)]


def human_format(num, precision=5):
    s = "{:." + str(precision) + "g}"
    num = float(s.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return f'{num:.{precision}f}{" KMBT"[magnitude]}'


def put(value, filename):
    persist_dir.mkdir(exist_ok=True)
    filename = persist_dir / filename
    print("Saving to ", filename)
    joblib.dump(value, filename)


def get(filename):
    filename = persist_dir / filename
    assert filename.exists(), f"{filename} doesn't exist"
    print("Loading from ", filename)
    return joblib.load(filename)


def smooth(arr, window):
    return pd.Series(arr).rolling(window, min_periods=1).mean().values



def log_sum_weighted_exp(val1, val2, weight1, weight2):
    val_max = np.where(val1 > val2, val1, val2)
    val1_exp = weight1 * np.exp(val1 - val_max)
    val2_exp = weight2 * np.exp(val2 - val_max)
    return val_max + np.log(val1_exp + val2_exp)




def adjust_lightness(color, amount=0.5):
    """ https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib """
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])



def seed_all(seed):
    """Seed all devices deterministically off of seed and somewhat
    independently."""
    np.random.seed(seed)
    random.seed(seed)




def ESS(x):
    """ Compute the effective sample size of estimand of interest. Vectorised implementation
        from: https://jwalton.info/Efficient-effective-sample-size-python/
     """
    if x.shape[0] > x.shape[1]:
        x = x.T

    m_chains, n_iters = x.shape

    def variogram(t):
        return ((x[:, t:] - x[:, :(n_iters - t)])**2).sum() / (m_chains * (n_iters - t))

    post_var = gelman_rubin(x)

    t = 1
    rho = np.ones(n_iters)
    negative_autocorr = False

    # Iterate until the sum of consecutive estimates of autocorrelation is negative
    while not negative_autocorr and (t < n_iters):
        rho[t] = 1 - variogram(t) / (2 * post_var)

        if not t % 2:
            negative_autocorr = sum(rho[t - 1:t + 1]) < 0

        t += 1

    return int(m_chains * n_iters / (1 + 2 * rho[1:t].sum()))


def ci(a, which=95, axis=None):
    """Return a percentile range from an array of values."""
    p = 50 - which / 2, 50 + which / 2
    return np.nanpercentile(a, p, axis)


def ESSl(lw):
    """ESS (Effective sample size) computed from log-weights.

    Parameters
    ----------
    lw: (N,) ndarray
        log-weights

    Returns
    -------
    float
        the ESS of weights w = exp(lw), i.e. the quantity
        sum(w**2) / (sum(w))**2

    Note
    ----
    The ESS is a popular criterion to determine how *uneven* are the weights.
    Its value is in the range [1, N], it equals N when weights are constant,
    and 1 if all weights but one are zero.

    """
    w = np.exp(lw - lw.max())
    return (w.sum())**2 / np.sum(w**2)


def gelman_rubin(x):
    """ Estimate the marginal posterior variance. Vectorised implementation. """
    m_chains, n_iters = x.shape

    # Calculate between-chain variance
    B_over_n = ((np.mean(x, axis=1) - np.mean(x))**2).sum() / (m_chains - 1)

    # Calculate within-chain variances
    W = ((x - x.mean(axis=1, keepdims=True))**2).sum() / (m_chains * (n_iters - 1))

    return W * (n_iters - 1) / n_iters + B_over_n


def get_unique_dir(comment=None):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    host = socket.gethostname()
    name = f"{current_time}_{host}"
    if comment:
        name = f"{name}_{comment}"
    return name


def spread(X, N, axis=0):
    """
    Takes a 1-d vector and spreads it out over
    N rows s.t spread(X, N).sum(0) = X
    """
    return (1 / N) * duplicate(X, N, axis)


def duplicate(X, N, axis=0):
    """
    Takes a 1-d vector and duplicates it across
    N rows s.t spread(X, N).sum(axis) = N*X
    """
    order = (N, 1) if axis == 0 else (1, N)
    return X.unsqueeze(axis).repeat(*order)


def safe_json_load(path):
    path = Path(path)
    res = {}
    try:
        if path.stat().st_size != 0:
            with open(path, encoding='utf-8') as data_file:
                res = json.load(data_file)
    except FileNotFoundError as _:
        print(f"{path} not found:")
        print("------------------------------")
    except json.JSONDecodeError as _:
        print(f"Error decoding JSON in {path}:")
        print("------------------------------")
    return res


# Safe initalizers


def numpyify(val):
    if isinstance(val, dict):
        return {k: np.array(v) for k, v in val.items()}
    if isinstance(val, (float, int, list, np.ndarray)):
        return np.array(val)
    else:
        raise ValueError("Not handled")


def array(val):
    return numpyify(val)


def slist(val):
    """
    safe list
    """
    return val if isinstance(val, list) else [val]


def notnan(val):
    return not pd.DataFrame(val).isnull().values.any()


def get_unique_legend(axes):
    unique = {}
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        for label, handle in zip(labels, handles):
            unique[label] = handle
    handles, labels = zip(*unique.items())
    return handles, labels


def get_all_dirs(path):
    return [p for p in Path(path).glob("*") if p.is_dir()]


def timeit(message=None):
    def decorator(method):
        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            if message:
                print(f'{message}: {te - ts:.2f}s')
            else:
                print(f'{method.__name__}: {te - ts:.2f}s')
            return result
        return timed
    return decorator


def get_frequency(y):
    y = np.bincount(y)
    ii = np.nonzero(y)[0]
    return dict(zip(ii, y[ii]))


def default_init(args):
    if isinstance(args, dict):
        args = SimpleNamespace(**args)
    seed_all(args.seed)
    args.home_dir = str(Path(args.home_dir).absolute())
    return args


def join_path(*args):
    return str(Path("/".join(args)))  # trick to remove multiple backslashes


def add_home(home_dir, *args):
    return [join_path(home_dir, p) for p in args]


# following https://martinheinz.dev/blog/50
@singledispatch
def to_np(val):
    return np.array(val)


@to_np.register
def _(val: dict) -> Any:
    return {k: np.array(v) for k, v in val.items()}


def run_bash_command(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True, check=True)
    return result.stdout.strip()


def str_to_py(str_or_py):
    return literal_eval(str_or_py) if isinstance(str_or_py, str) else str_or_py


def fast_csv_sample(csv_path):
    """
    this ugly bash command allows us to sample a csv row w/o actually iterating through the large csv
    15x faster pandas
    path = Path(data_dir) / '2024-01-26-locarno/9/63/permutes.csv'
    csv_sample(path)
    --------------------------------------------------
    CPU times: user 3.75 s, sys: 346 ms, total: 4.09 s
    Wall time: 4.09 s
    --------------------------------------------------


    fast_csv_sample(path)
    --------------------------------------------------
    CPU times: user 0 ns, sys: 3.29 ms, total: 3.29 ms
    Wall time: 273 ms
    --------------------------------------------------
    """
    csv_path = str(csv_path)
    command = f"grep -m 1 \"^$(shuf -i 0-$(tail -n 1 {csv_path} | grep -o '^[^,]*') -n 1),\" {csv_path}"
    result = run_bash_command(command)
    return [str_to_py(r) for r in str_to_py(result)[1:]]
