from __future__ import annotations
from __future__ import division, print_function

import time
from collections import defaultdict, deque
from datetime import timedelta

from typing import Any
import numpy as np
import torch
from torch import inf
from tqdm import tqdm


class Meter(object):
    """
    A class to track and compute statistics over a series of values.

    Attributes:
        window_size (int): The maximum number of recent values to consider for statistics.
        fmt (str): Format string for printing the meter's statistics.
        deque (collections.deque): A deque holding the most recent values added.
        total (float): The sum of all values added.
        count (int): The total number of values added.
        mean (float): The current mean of the values.

    Methods:
        reset(): Resets the meter to its initial state.
        update(value): Adds a new value to the meter, updating all statistics.
        var: Returns the variance of the values.
        sample_var: Returns the sample variance of the values.
        median: Returns the median of the values.
        smoothed_avg: Returns the average of the values in the deque.
        avg: Returns the average of all values added.
        global_avg: Returns the average of all values added (same as avg).
        max: Returns the maximum value in the deque.
        min: Returns the minimum value in the deque.
        value: Returns the most recently added value.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({avg:.4f})"
        self.deque = deque(maxlen=window_size)
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

    def step(self, value):
        self.update(value)

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
        return self.avg

    @property
    def max(self):
        return max(self.deque)

    @property
    def min(self):
        return min(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.smoothed_avg,
            global_avg=self.global_avg,
            max=self.max,
            min=self.min,
            value=self.value)


class MetricLogger(object):
    """
    A utility class for logging and tracking metrics during training or evaluation.

    This class supports logging scalar values (e.g., loss, accuracy) and uses a sliding window
    to compute statistics (e.g., average, median) over recent values. It can also log metrics
    to Weights & Biases (wandb) if provided.

    Attributes:
        meters (defaultdict): A collection of Meter objects for tracking different metrics.
        print_freq (int): Frequency (in iterations) at which to print logged metrics.
        header (str): A header string to prepend to log messages.
        wandb: An optional Weights & Biases logging object for remote tracking.
        delimiter (str): The delimiter to use when joining multiple metric strings for printing.

    Args:
        header (str, optional): A header string to prepend to log messages. Defaults to ''.
        print_freq (int, optional): Frequency (in iterations) at which to print logged metrics. Defaults to 1.
        wandb (optional): An optional Weights & Biases logging object for remote tracking. Defaults to None.
        window_size (int, optional): The size of the sliding window for computing statistics. Defaults to 20.
        fmt (str, optional): A format string for printing each metric. Defaults to None.

    Methods:
        update(**kwargs): Updates the tracked metrics with new values.
        add_meter(name, meter): Adds a new Meter object for tracking a specific metric.
        step(iterable): Iterates over an iterable, logging metrics at the specified frequency and computing total time.
    """

    def __init__(self, header='', print_freq=1, wandb=None, window_size=20, fmt=None):
        self.meters = defaultdict(lambda: Meter(window_size=window_size, fmt=fmt))
        self.print_freq = print_freq
        self.header = header
        self.wandb = wandb
        self.delimiter = " "

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int)), f'{k} is of type {type(v)}'
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

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __str__(self):
        loss_str = [f"{name}: {str(meter)}" for name, meter in self.meters.items()]
        return self.delimiter.join(loss_str)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def step(self, iterable):
        iterable = list(iterable)  # unnest if generator
        start_time = time.time()
        end = time.time()
        iter_time = Meter(fmt='{avg:.4f}')
        data_time = Meter(fmt='{avg:.4f}')
        space_fmt = f':{len(str(len(iterable)))}d'
        pbar = tqdm(total=len(iterable)) if self.use_tqdm else None  # Initialize tqdm progress bar
        for i, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            eta_seconds = iter_time.global_avg * (len(iterable) - i)
            eta_string = str(timedelta(seconds=int(eta_seconds)))
            log_msg = self.delimiter.join([
                self.header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
            log_data = log_msg.format(
                i, len(iterable), eta=eta_string,
                meters=str(self),
                time=str(iter_time), data=str(data_time)
            )
            if pbar:
                pbar.set_description(log_data)  # Update tqdm description
                pbar.update(1)  # Update tqdm progress
            else:
                print(log_data)  # Print log data if not using tqdm
            end = time.time()
        if self.use_tqdm:
            pbar.close()  # Close tqdm progress bar
        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        print(f'{self.header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)')


class ConvergenceMeter(object):
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

    def __init__(self, mode='min', patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, eps=1e-8):

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
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def update(self, metrics, epoch=None):
        self.step(metrics, epoch=epoch)

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

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError(f'mode {mode} is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError(f'threshold mode {threshold_mode} is unknown!')

        self.mode_worse = inf if mode == 'min' else -inf
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode


class BestMeter(object):
    """ This is like ConvergenceMeter except it stores the
        best result in a set of results. To be used in a
        grid search

    Args:
        mode (str): One of `min`, `max`. In `min` mode, best will
            be updated when the quantity metered is lower than the current best;
            in `max` mode best will be updated when the quantity metered is higher
            than the current best. Default: 'max'.

    """

    def __init__(self, name='value', mode='max', object_name='epoch', verbose=True):

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
        return a < best if self.mode == 'min' else a > best

    def _init_is_better(self, mode):
        if mode not in {'min', 'max'}:
            raise ValueError(f'mode {mode} is unknown!')
        self.mode_worse = inf if mode == 'min' else -inf
        self.mode = mode
