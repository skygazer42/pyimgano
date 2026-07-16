"""Small argparse compatibility helpers."""

from __future__ import annotations

import argparse
from typing import Any, Sequence


class _BooleanOptionalAction(argparse.Action):
    """Python 3.8 fallback for argparse.BooleanOptionalAction."""

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        default: Any = None,
        type: Any = None,
        choices: Any = None,
        required: bool = False,
        help: str | None = None,
        metavar: str | None = None,
    ) -> None:
        expanded_options = []
        for option_string in option_strings:
            expanded_options.append(option_string)
            if option_string.startswith("--"):
                expanded_options.append(f"--no-{option_string[2:]}")

        super().__init__(
            option_strings=expanded_options,
            dest=dest,
            nargs=0,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
        )

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        del parser, values
        setattr(namespace, self.dest, not str(option_string).startswith("--no-"))

    def format_usage(self) -> str:
        return " | ".join(self.option_strings)


BooleanOptionalAction = getattr(
    argparse,
    "BooleanOptionalAction",
    _BooleanOptionalAction,
)

__all__ = ["BooleanOptionalAction"]
