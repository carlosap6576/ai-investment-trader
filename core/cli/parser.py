"""
Shared CLI utilities for argument parsing.

Provides FriendlyArgumentParser used by all strategy CLIs.
"""

import argparse
import sys


class FriendlyArgumentParser(argparse.ArgumentParser):
    """
    Custom ArgumentParser that shows friendly help on errors.

    Instead of just printing "error: invalid argument", this parser
    shows helpful usage examples and valid options.
    """

    def __init__(self, *args, quick_usage=None, examples=None, valid_options=None, **kwargs):
        """
        Initialize the parser with optional help customization.

        Args:
            quick_usage: Quick usage string to show on error.
            examples: List of example command strings.
            valid_options: Dict of option_name -> list of valid values.
        """
        super().__init__(*args, **kwargs)
        self.quick_usage = quick_usage or "python <script> -s SYMBOL [OPTIONS]"
        self.examples = examples or []
        self.valid_options = valid_options or {}

    def error(self, message):
        """Override error to show helpful information."""
        sys.stderr.write(f"\n{'='*70}\n")
        sys.stderr.write(f"ERROR: {message}\n")
        sys.stderr.write(f"{'='*70}\n\n")

        sys.stderr.write("QUICK USAGE:\n")
        sys.stderr.write(f"  {self.quick_usage}\n\n")

        if self.examples:
            sys.stderr.write("EXAMPLES:\n")
            for example in self.examples[:3]:
                sys.stderr.write(f"  {example}\n")
            sys.stderr.write("\n")

        if self.valid_options:
            for name, values in self.valid_options.items():
                sys.stderr.write(f"VALID {name.upper()}:\n")
                sys.stderr.write(f"  {', '.join(values)}\n\n")

        sys.stderr.write(f"For full help, run: {sys.argv[0]} --help\n\n")
        sys.exit(2)


def check_help_flag(help_text_func):
    """
    Check for help flag and print help if found.

    Args:
        help_text_func: Function that returns the help text string.
    """
    if '-h' in sys.argv or '--help' in sys.argv or len(sys.argv) == 1:
        print(help_text_func())
        sys.exit(0)
