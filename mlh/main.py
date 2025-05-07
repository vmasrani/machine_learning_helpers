from parallel import pmap
from time import sleep


def sleep_1s(x):
    sleep(1)
    return x


def main():
    print("Hello from machine-learning-helpers!")
    pmap(sleep_1s, [1, 2, 3])

if __name__ == "__main__":
    main()
