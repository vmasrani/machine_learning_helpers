
from __future__ import division, print_function
import argparse
import inspect
import contextlib
import logging
import random
import ssl
import time
import warnings
from collections import defaultdict, deque
from datetime import timedelta
from pathlib import Path

import numpy as np
import requests
import torch
from typing import Any, Deque
from torch import inf
from urllib3.exceptions import InsecureRequestWarning

from torch import inf

persist_dir = Path('./.persistdir')
get_unix_time = lambda x: int(time.mktime(x.timetuple()))
# import flavor

old_merge_environment_settings = requests.Session.merge_environment_settings
persist_dir = Path("./.persistdir")

ssl._create_default_https_context = ssl._create_unverified_context

logging.getLogger("urllib3").setLevel(logging.ERROR)


@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        # Verification happens only once per connection so we need to close
        # all the opened adapters once we're done. Otherwise, the effects of
        # verify=False persist beyond the end of this context manager.
        opened_adapters.add(self.get_adapter(url))

        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings["verify"] = False

        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings

        for adapter in opened_adapters:
            with contextlib.suppress(Exception):
                adapter.close()

def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_argument(parser, name, value) -> None:
    if isinstance(value, bool):
        parser.add_argument(f"--{name}", default=value, type=str2bool)
    elif isinstance(value, list):
        parser.add_argument(f"--{name}", default=value, type=type(value[0]), nargs='+')
    elif isinstance(value, (int, float, str)):
        parser.add_argument(f"--{name}", default=value, type=type(value))
    else:
        raise ValueError(f"Unknown type {type(value)} for {name}")


class HyperParams:
    """
    A class for parsing command line arguments.

    This class uses the `argparse` module to parse command line arguments and set them as attributes of the class instance.
    The default values for the attributes are taken from the class attributes.

    Example usage:
    ```
    class MyArgs(HyperParams):
        arg1 = False
        arg2 = 'cat'
        arg3 = [1, 2, 3]
        new_param: int # fill inline


    myargs = MyArgs()
    print(myargs)
    print(myargs.to_dict())
    myargs.new_param = 100
    print(myargs)
    ```
    This will create a `MyArgs` instance with the default values for the attributes, and print them in a pretty format.
    """

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        for name, value in self._get_members():
            add_argument(self.parser, name, value)
        self.parse_args()

    def parse_args(self, args=None) -> None:
        args, argv = self.parser.parse_known_args(args)
        # crude catches here to transition away from sacred, clean up later
        assert "with" not in argv, "Still using Sacred format"

        # Remove --unobserved from argv
        if "--unobserved" in argv:
            argv.remove("--unobserved")

        # Raise an assertion if there are any remaining items in argv
        assert len(argv) == 0, f"Unexpected command line arguments: {argv}"
        for name, value in vars(args).items():
            setattr(self, name, value)
        del self.parser

    def __str__(self):
        return self._str_helper(0)

    def _get_members(self):
        for name, value in inspect.getmembers(self):
            if (
                not name.startswith("__")
                and not inspect.ismethod(value)
                and name != 'parser'
            ):
                yield name, value

    def to_dict(self):
        """ return a dict representation of the config """
        return {k: v.to_dict() if isinstance(v, HyperParams) else v for k, v in self.__dict__.items()}

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        parts = []
        for k, v in self.__dict__.items():
            if k == "parser":
                continue
            if isinstance(v, HyperParams):
                parts.extend(("%s:\n" % k, v._str_helper(indent + 1))) # pylint: disable=W0212
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts).strip()


# =====================
#   Loggers and Meters
# =====================


# from the excellent https://github.com/pytorch/vision/blob/master/references/detection/utils.py
class Meter:
    """Track a series of values and provide access to a number of metric"""

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque: Any = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

        self.M2 = 0
        self.mean = 0
        self.fmt = fmt

    def reset(self):
        self.total = 0.0
        self.count = 0
        self.M2 = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

    @property
    def var(self):
        return self.M2 / self.count if self.count > 2 else 0

    @property
    def sample_var(self):
        return self.M2 / (self.count - 1) if self.count > 2 else 0

    @property
    def median(self):
        return np.median(self.deque)

    @property
    def smoothed_avg(self):
        return np.mean(self.deque)

    @property
    def avg(self):
        return self.total / self.count

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.smoothed_avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter=" ", header="", print_freq=1, window_size=20, wandb=None):
        print(f"Window size: {window_size}")
        self.meters = defaultdict(lambda: Meter(window_size=window_size))
        self.delimiter = delimiter
        self.print_freq = print_freq
        self.header = header
        self.wandb = wandb

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int)), f"{k} is of type {type(v)}"
            self.meters[k].update(v)
        if self.wandb is not None:
            self.wandb.log(kwargs)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def __str__(self):
        loss_str = [f"{name}: {str(meter)}" for name, meter in self.meters.items()]
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def step(self, iterable):
        start_time = time.time()
        end = time.time()
        iter_time = Meter(fmt="{avg:.4f}")
        data_time = Meter(fmt="{avg:.4f}")
        space_fmt = f":{len(str(len(iterable)))}d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    self.header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    self.header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for i, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % self.print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                self.header, total_time_str, total_time / len(iterable)
            )
        )


class ConvergenceMeter:
    """This is a modification of pytorch's ReduceLROnPlateau object
        (https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau)
        which acts as a convergence meter. Everything
        is the same as ReduceLROnPlateau, except it doesn't
        require an optimizer and doesn't modify the learning rate.
        When meter.converged(loss) is called it returns a boolean that
        says if the loss has converged.

    Args:
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity metered has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity metered has stopped increasing. Default: 'min'.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.

    Example:
        >>> meter = Meter('min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     if meter.converged(val_loss):
        >>>         break
    """

    def __init__(
        self,
        mode="min",
        patience=10,
        verbose=False,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        eps=1e-8,
    ):
        self.has_converged = False
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = 0
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def update(self, metrics, epoch=None):
        return self.step(metrics, epoch=None)

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self.has_converged = True

        return self.has_converged

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError(f"mode {mode} is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError(f"threshold mode {threshold_mode} is unknown!")

        self.mode_worse = inf if mode == "min" else -inf
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode


class BestMeter:
    """This is like ConvergenceMeter except it stores the
        best result in a set of results. To be used in a
        grid search

    Args:
        mode (str): One of `min`, `max`. In `min` mode, best will
            be updated when the quantity metered is lower than the current best;
            in `max` mode best will be updated when the quantity metered is higher
            than the current best. Default: 'max'.

    """

    def __init__(self, name="value", mode="max", object_name="epoch", verbose=True):
        self.has_converged = False
        self.verbose = verbose
        self.mode = mode
        self.name = name
        self.obj_name = object_name
        self.best = None
        self.best_obj = None
        self.mode_worse = None  # the worse value for the chosen mode
        self._init_is_better(mode=mode)
        self._reset()

    def _reset(self):
        self.best = self.mode_worse

    def step(self, metrics, **kwargs):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)

        if self.is_better(current, self.best):
            self.best = current
            self.best_obj = kwargs
            if self.verbose:
                print("*********New best**********")
                print(f"{self.name}: ", current)
                print(f"{self.best_obj}")
                print("***************************")
            return True

        return False

    def is_better(self, a, best):
        return a < best if self.mode == "min" else a > best

    def _init_is_better(self, mode):
        if mode not in {"min", "max"}:
            raise ValueError(f"mode {mode} is unknown!")
        self.mode_worse = inf if mode == "min" else -inf
        self.mode = mode


def detect_cuda(args):
    if "cuda" not in args.__dict__:
        return args
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.cuda = True
    else:
        args.device = torch.device("cpu")
        args.cuda = False
    return args


def seed_all(seed):
    """Seed all devices deterministically off of seed and somewhat
    independently."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# from https://stackoverflow.com/questions/50246304/using-python-decorators-to-retry-request
def retry(times, exceptions, delay=1):
    """
    Retry Decorator
    Retries the wrapped function/method `times` times if the exceptions listed
    in ``exceptions`` are thrown
    :param times: The number of times to repeat the wrapped function/method
    :type times: Int
    :param Exceptions: Lists of exceptions that trigger a retry attempt
    :type Exceptions: Tuple of Exceptions

    Example:
    @retry(times=3, exceptions=(ValueError, TypeError))
    def foo1():
        print('Some code here ....')
        print('Oh no, we have exception')
        raise ValueError('Some error')

    foo1()
    """
    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    print(
                        'Exception thrown when attempting to run %s, attempt '
                        '%d of %d' % (func, attempt, times)
                    )
                    print(e)
                    attempt += 1
                    time.sleep(delay)
            return func(*args, **kwargs)
        return newfn
    return decorator


def default_init(args):
    seed_all(args.seed)
    args = detect_cuda(args)
    args.home_dir = str(Path(args.home_dir).absolute())
    print("-" * 37, "hypers", "-" * 37)
    print(args)
    print("-" * 80)
    return args


def join_path(*args):
    return str(Path("/".join(args)))  # trick to remove multiple backslashes


def add_home(home_dir, *args) -> Any:
    if len(args) == 1:
        return join_path(home_dir, args[0])
    return [join_path(home_dir, p) for p in args]
