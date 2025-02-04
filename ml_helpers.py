from __future__ import division, print_function

import colorsys
import contextlib
import json
import os
import random
import socket
import ssl
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from functools import singledispatch
from pathlib import Path
from pprint import pprint
from types import SimpleNamespace
from typing import Any

import janitor
import joblib
import matplotlib.colors as mc
import numpy as np
import pandas as pd
import requests
import torch
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

persist_dir = Path('./.persistdir')

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_str_to_json(data):
    if isinstance(data, list):
        return [parse_str_to_json(item) for item in data]
    data = data.replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false")
    return json.loads(data)



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


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


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


def get_data_loader(dataset, batch_size, args, shuffle=True):
    """Args:
        np_array: shape [num_data, data_dim]
        batch_size: int
        device: torch.device object

    Returns: torch.utils.data.DataLoader object
    """

    if args.device == torch.device('cpu'):
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {}

    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def split_train_test_by_percentage(dataset, train_percentage=0.8):
    """ split pytorch Dataset object by percentage """
    train_length = int(len(dataset) * train_percentage)
    return torch.utils.data.random_split(dataset, (train_length, len(dataset) - train_length))


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


def detect_cuda(args):
    if "cuda" not in args.__dict__:
        return args
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.cuda = True
    else:
        args.device = torch.device('cpu')
        args.cuda = False
    return args


def log_sum_weighted_exp(val1, val2, weight1, weight2):
    val_max = np.where(val1 > val2, val1, val2)
    val1_exp = weight1 * np.exp(val1 - val_max)
    val2_exp = weight2 * np.exp(val2 - val_max)
    return val_max + np.log(val1_exp + val2_exp)


def logaddexp(a, b):
    """Returns log(exp(a) + exp(b))."""

    return torch.logsumexp(torch.cat([a.unsqueeze(0), b.unsqueeze(0)]), dim=0)


def lognormexp(values, dim=0):
    """Exponentiates, normalizes and takes log of a tensor.
    """

    log_denominator = torch.logsumexp(values, dim=dim, keepdim=True)
    # log_numerator = values
    return values - log_denominator


def make_sparse(sparse_mx, args):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)

    indices = tensor(np.vstack((sparse_mx.row, sparse_mx.col)), args, torch.long)
    values = tensor(sparse_mx.data, args)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adjust_lightness(color, amount=0.5):
    """ https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib """
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def exponentiate_and_normalize(values, dim=0):
    """Exponentiates and normalizes a tensor.

    Args:
        values: tensor [dim_1, ..., dim_N]
        dim: n

    Returns:
        result: tensor [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =
                            exp(values[i_1, ..., i_N])
            ------------------------------------------------------------
             sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    return torch.exp(lognormexp(values, dim=dim))


def seed_all(seed):
    """Seed all devices deterministically off of seed and somewhat
    independently."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_grads(model):
    return torch.cat([torch.flatten(p.grad.clone()) for p in model.parameters()]).cpu()


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

def tensor(data, args=None, dtype=torch.float, device=torch.device('cpu')):
    if args is not None:
        device = args.device
    if torch.is_tensor(data):
        return data.to(dtype=dtype, device=device)
    elif isinstance(data, list) and torch.is_tensor(data[0]):
        return torch.stack(data)
    else:
        return torch.tensor(np.array(data), device=device, dtype=dtype)


def parameter(*args, **kwargs):
    return torch.nn.Parameter(tensor(*args, **kwargs))


def numpyify(val):
    if isinstance(val, dict):
        return {k: np.array(v) for k, v in val.items()}
    if isinstance(val, (float, int, list, np.ndarray)):
        return np.array(val)
    if isinstance(val, (torch.Tensor)):
        return val.cpu().numpy()
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


def timeit(display=True):
    def decorator(method):
        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            if display:
                print(f'{method.__name__}:  {te - ts} s')
                return result
            return result, method.__name__, te - ts
        return timed
    return decorator


def get_frequency(y):
    y = np.bincount(y)
    ii = np.nonzero(y)[0]
    return dict(zip(ii, y[ii]))


def get_debug_args():
    args = SimpleNamespace()
    args.model_dir = './models'
    args.data_dir = ''

    # Training settings
    args.epochs = 10
    args.seed = 0
    args.cuda = True
    args.warmup = 5000
    args.lr_max = 0.00005
    args.eval_steps = 4
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args


def default_init(args):
    if isinstance(args, dict):
        args = SimpleNamespace(**args)
    seed_all(args.seed)
    args = detect_cuda(args)
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


@to_np.register
def _(val: torch.Tensor):
    return val.cpu().numpy()
