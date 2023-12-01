import argparse
import inspect
import os
import sys
from copy import deepcopy


class Hypers:
    """
    A class for parsing command line arguments.

    This class uses the `argparse` module to parse command line arguments
    and set them as attributes of the class instance.
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
    This will create a `MyArgs` instance with the default values for the attributes,
    and print them in a pretty format.
    """
    # tuples and dicts defined here won't be iterated over
    valid_types = (int, float, bool, str, list)
    cmdline_args = tuple(s.replace("--", '').split("=")[0] for s in sys.argv[1:] if "=" in s)
    parser = argparse.ArgumentParser()
    cmdline_parser = deepcopy(parser)
    args_color_map = {
        'default': 34,
        'config': 35,
        'command_line': 33
    }
    args_list = {
        'default': [],
        'config': [],
        'command_line': []
    }

    file_list = {}

    def __init__(self) -> None:
        for name, value in self._get_members():
            self._add_argument(self.parser, name, value)
            if name in self.cmdline_args:
                self._add_argument(self.cmdline_parser, name, value)
        self.parse_args()
        print(self)

    def _str2bool(self, v) -> bool:
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def _add_argument(self, parser, name, value) -> None:
        if isinstance(value, bool):
            parser.add_argument(f"--{name}", default=value, type=self._str2bool)
        elif isinstance(value, list):
            # assumes list contains entries of same type
            parser.add_argument(f"--{name}", default=value, type=(type(value[0]) if len(value) > 0 else int), nargs='+')
        elif isinstance(value, (int, float, str)):
            parser.add_argument(f"--{name}", default=value, type=type(value))
        else:
            raise ValueError(f"Unknown type {type(value)} for {name}")

    def _parse_config_file(self, file):
        variables = {}
        if "--" in file:
            raise ValueError(f"{file} is not a valid argument.")

        if not os.path.isfile(file):
            raise ValueError(f"{file} is not a valid file.")

        with open(file) as f:
            exec(f.read(), variables)

        count = 0
        for name, value in variables.items():
            if name.startswith('_'):
                continue
            setattr(self, name, value)
            self.args_list['config'].append(name)
            count += 1
        self.file_list[file] = count

    def __str__(self):
        return self._str_helper(0)

    def _get_members(self):
        def member_filter(x): return isinstance(x, self.valid_types) and x != "__main__"  # catch main manually
        yield from inspect.getmembers(self, member_filter)

    def parse_args(self, args=None) -> None:
        # need to parse in this order:
        # 1. default
        # 2. configs
        # 3. command line
        # this loads default args + command line args
        default_args, argv = self.parser.parse_known_args(args)
        cmdline_args, argv = self.cmdline_parser.parse_known_args(args)

        self.args_list['default'] = list(default_args.__dict__.keys())
        self.args_list['command_line'] = list(cmdline_args.__dict__.keys())

        # 1. load default args
        for name, value in vars(default_args).items():
            setattr(self, name, value)

        # handle special cmd line args here
        # Remove --unobserved from argv (used by wandb)
        if "--unobserved" in argv:
            argv.remove("--unobserved")

        # 2. load config args
        for config in argv:
            self._parse_config_file(config)

        # 3. load cmd line args
        for name, value in vars(cmdline_args).items():
            setattr(self, name, value)

    def _color_text(self, test_str, ansi_code=None):
        # see https://www.geeksforgeeks.org/how-to-add-colour-to-text-python/
        if ansi_code is None:
            if test_str in self.args_list['command_line']:
                ansi_code = self.args_color_map['command_line']
            elif test_str in self.args_list['config']:
                ansi_code = self.args_color_map['config']
            else:
                ansi_code = self.args_color_map['default']

        def _helper(code):
            return f"\33[{code}m"
        return _helper(ansi_code) + test_str + _helper(0)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        default = self._color_text('default', self.args_color_map['default'])
        config = self._color_text('config', self.args_color_map['config'])
        command_line = self._color_text('command_line', self.args_color_map['command_line'])
        legend = f" ({default}, {config}, {command_line})"
        parts = ["-" * 40 + "HyperParams" + "-" * 40 + "\n"]
        parts += [" " * 28 + legend + " " * 30 + "\n"]
        parts += ["\n"]
        for file, count in self.file_list.items():
            parts += [f"Reading {count} arguments from {file}" + "\n"]
        parts += ["\n"]
        parts.extend(
            "%s: %s\n" % (self._color_text(k), v) for k, v in self.__dict__.items()
        )
        parts = [' ' * (indent * 4) + p for p in parts]
        parts += ["-" * len(parts[0]) + "\n"]
        return "".join(parts).strip()

    def to_dict(self):
        """ return a dict representation of the config """
        return dict(self.__dict__.items())

    def merge_from_dict(self, d):
        self.__dict__.update(d)
