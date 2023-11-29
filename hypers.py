import argparse
import inspect


class HyperParams:
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

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        for name, value in self._get_members():
            self._add_argument(name, value)
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

    def _color_text(self, str, ansi_code=35):
        # see https://www.geeksforgeeks.org/how-to-add-colour-to-text-python/
        def _helper(code):
            return f"\33[{code}m"
        return _helper(ansi_code) + str + _helper(0)

    def _add_argument(self, name, value) -> None:
        if isinstance(value, bool):
            self.parser.add_argument(f"--{name}", default=value, type=self._str2bool)
        elif isinstance(value, list):
            self.parser.add_argument(f"--{name}", default=value, type=(type(value[0]) if len(value) > 0 else int), nargs='+')

        elif isinstance(value, (int, float, str)):
            self.parser.add_argument(f"--{name}", default=value, type=type(value))
        else:
            raise ValueError(f"Unknown type {type(value)} for {name}")

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
        return {k: v.to_dict() if isinstance(v, HyperParams) else v
                for k, v in self.__dict__.items()}

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        parts = ["-" * 40 + "HyperParams" + "-" * 40 + "\n"]
        for k, v in self.__dict__.items():
            if k == "parser":
                continue
            if isinstance(v, HyperParams):
                parts.extend(("%s:\n" % self._color_text(k), v._str_helper(indent + 1)))
            else:
                parts.append("%s: %s\n" % (self._color_text(k), v))
        parts = [' ' * (indent * 4) + p for p in parts]
        parts += ["-" * len(parts[0]) + "\n"]
        return "".join(parts).strip()
