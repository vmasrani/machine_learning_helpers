import argparse
import inspect
import os
import sys
from copy import deepcopy

parser = argparse.ArgumentParser()
cmdline_parser = deepcopy(parser)

VALID_TYPES = (int, float, bool, str, list)
CMDLINE_ARGS = tuple(s.replace("--", '').split("=")[0] for s in sys.argv[1:] if "=" in s)
COMMAND_LINE_COLOR = 33
DEFAULT_COLOR      = 34
CONFIG_COLOR       = 35

COMMAND_LINE_ARGS = []
DEFAULT_ARGS      = []
CONFIG_ARGS       = []

FILE_LIST = {}

# see https://www.geeksforgeeks.org/how-to-add-colour-to-text-python/
def color(test_str, ansi_code):
    return f"\33[{ansi_code}m{test_str}\33[0m"

HEADER = f"""\
{'-' * 40}HyperParams{'-' * 40}
{' ' * 25}(color code: \
{color('default', DEFAULT_COLOR)}, \
{color('config',  CONFIG_COLOR)}, \
{color('command_line', COMMAND_LINE_COLOR)})\
{' ' * 30}
"""

FOOTER = f"{'-' * 90}\n"

def get_code(test_str):
    if test_str in COMMAND_LINE_ARGS:
        return COMMAND_LINE_COLOR
    elif test_str in CONFIG_ARGS:
        return CONFIG_COLOR
    else:
        return DEFAULT_COLOR

def add_argument(parser, name, value) -> None:
    if isinstance(value, bool):
        parser.add_argument(f"--{name}", default=value, type=str2bool)
    elif isinstance(value, list):
        assert all(isinstance(i, type(value[0])) for i in value), f"All elements in {name} must be of the same type"
        parser.add_argument(f"--{name}", default=value, type=(type(value[0]) if len(value) > 0 else int), nargs='+')
    elif isinstance(value, (int, float, str)):
        parser.add_argument(f"--{name}", default=value, type=type(value))
    else:
        raise ValueError(f"Unknown type {type(value)} for {name}")

def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def member_filter(x):
    return isinstance(x, VALID_TYPES) and x != "__main__"  # catch main manually

def read_config(file):
    variables = {}
    if "--" in file:
        raise ValueError(f"{file} is not a valid argument.")

    if not os.path.isfile(file):
        raise ValueError(f"{file} is not a valid file.")

    with open(file) as f:
        exec(f.read(), variables)

    return {k:v for k, v in variables.items() if not k.startswith('_')}



"""
A class for parsing command line arguments.

This class uses the `argparse` module to parse command line arguments
and set them as attributes of the class instance.
The default values for the attributes are taken from the class attributes.

Example usage:

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

This will create a `MyArgs` instance with the default values for the attributes,
and print them in a pretty format.
"""
class Hypers:
    def __init__(self) -> None:
        for name, value in self._get_members():
            add_argument(parser, name, value)
            if name in CMDLINE_ARGS:
                add_argument(cmdline_parser, name, value)
        self.parse_args()
        print(self)

    def __str__(self):
        files = [f"- Reading {count} arguments from {file}\n" for file, count in FILE_LIST.items()]
        args = [f"{color(k, get_code(k))}: {v}\n" for k, v in self.__dict__.items()]
        return "".join([HEADER, *files, *args, FOOTER])

    def _get_members(self):
        yield from inspect.getmembers(self, member_filter)

    def _load_default_args(self, default_args):
        for name, value in vars(default_args).items():
            setattr(self, name, value)

    def _handle_special_args(self, argv):
        if "--unobserved" in argv:
            argv.remove("--unobserved")

    def _load_config_args(self, argv):
        for config in argv:
            self._parse_config_file(config)

    def _parse_config_file(self, file):
        variables = read_config(file)
        for name, value in variables.items():
            setattr(self, name, value)
            CONFIG_ARGS.append(name)
        FILE_LIST[file] = len(variables)

    def _load_cmdline_args(self, cmdline_args):
        for name, value in vars(cmdline_args).items():
            setattr(self, name, value)

    def parse_args(self, args=None) -> None:
        default_args, argv = parser.parse_known_args(args)
        cmdline_args, argv = cmdline_parser.parse_known_args(args)

        DEFAULT_ARGS.extend(default_args.__dict__.keys())
        COMMAND_LINE_ARGS.extend(cmdline_args.__dict__.keys())

        self._load_default_args(default_args)
        self._handle_special_args(argv)
        self._load_config_args(argv)
        self._load_cmdline_args(cmdline_args)

    def to_dict(self):
        """ return a dict representation of the config """
        return dict(self.__dict__.items())

    def merge_from_dict(self, d):
        self.__dict__.update(d)
