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

from __future__ import print_function, division, absolute_import
from . import *
from . import log2
import sys
import unicodedata as ucd

if sys.version_info[0] < 3:
    if sys.maxunicode < 0x10FFFF:
        # workarounds for Python 2 "narrow" builds with UCS2-only support.

        _narrow_unichr = unichr

        def unichr(i):
            try:
                return _narrow_unichr(i)
            except ValueError:
                try:
                    padded_hex_str = hex(i)[2:].zfill(8)
                    escape_str = "\\U" + padded_hex_str
                    return escape_str.decode("unicode-escape")
                except UnicodeDecodeError:
                    raise ValueError("unichr() arg not in range(0x110000)")

    chr = unichr


def main(args=sys.argv):
    if len(args) == 1:
        print("usage: packTab [--rust] [--unsafe] data...")
        return 1

    language = "c"
    if args[1] == "--rust":
        language = "rust"
        args = args[1:]

    unsafe = False
    if args[1] == "--unsafe":
        unsafe = True
        args = args[1:]

    data = [int(v) for v in args[1:]]
    default = 0
    compression = 1

    solution = pack_table(data, default, compression=compression)

    lang = languages[language]
    lang.unsafe_array_access = unsafe

    code = Code("data")
    expr = solution.genCode(code, "get", language=lang, private=False)
    code.print_code(language=lang)

    return 0


if __name__ == "__main__":
    sys.exit(main())
