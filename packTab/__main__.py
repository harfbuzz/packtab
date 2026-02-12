# Copyright 2019 Facebook Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import *
import argparse
import sys


def main(args=None):
    parser = argparse.ArgumentParser(
        prog="packTab",
        description="Pack a list of integers into compact lookup tables.",
    )
    parser.add_argument(
        "data", nargs="+", type=int, help="integer data values to pack"
    )
    parser.add_argument(
        "--language",
        choices=["c", "rust"],
        default="c",
        help="output language (default: c)",
    )
    # Keep --rust as a shorthand for --language=rust.
    parser.add_argument(
        "--rust", action="store_true", help="shorthand for --language=rust"
    )
    parser.add_argument(
        "--unsafe",
        action="store_true",
        help="use unsafe array access (Rust only)",
    )
    parser.add_argument(
        "--default",
        type=int,
        default=0,
        help="default value for out-of-range indices (default: 0)",
    )
    parser.add_argument(
        "--compression",
        type=float,
        default=1,
        help="size vs speed tradeoff; higher = smaller tables (default: 1)",
    )
    parser.add_argument(
        "--name",
        default="data",
        help="namespace prefix for generated symbols (default: data)",
    )

    parsed = parser.parse_args(args)

    language = "rust" if parsed.rust else parsed.language

    solution = pack_table(
        parsed.data, parsed.default, compression=parsed.compression
    )

    lang = languageClasses[language](unsafe_array_access=parsed.unsafe)

    code = Code(parsed.name)
    solution.genCode(code, "get", language=lang, private=False)
    code.print_code(language=lang)

    return 0


if __name__ == "__main__":
    sys.exit(main())
