from time import sleep
import pandas as pd
from mlh.parallel import pmap
from mlh import flavor


def sleep_1s(x):
    sleep(1)
    return x


pmap(sleep_1s, range(10))


df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})



