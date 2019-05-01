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
					raise ValueError('unichr() arg not in range(0x110000)')
	chr = unichr


def print_solution(solution):
	print()
	functions, arrays, expr = solution.genCode('u')
	for (ret, name, args), body in functions.items():
		print('static inline %s %s (%s) { %s }' % (ret, name, args, body))
	print()
	for (elt, name), values in arrays.items():
		print('static const %s %s[%s] = {' % (elt, name, len(values)))
		w = max(len(str(v)) for v in values)
		n = 1 << int(log2(78 / (w + 1)))
		if (w + 2) * n <= 78:
			w += 1
		for i in range(0, len(values), n):
			line = values[i:i+n]
			print(' ', ','.join('%*s' % (w, v) for v in line))
		print('};')
	print()
	print('(void)', expr)
	print()


def solve(data, default=0):

	print("Unique values:", len(set(data)))
	solutions = pack_table(data, None, default).solutions

	print("All dominant solutions: (nLookups, nExtraOps, cost, fanOut)")
	for s in solutions:
		print(s)

	optimal = min(solutions, key=lambda s: s.nLookups * s.nLookups * s.fullCost)
	print("Optimal solution:", optimal)
	#print_solution(optimal)

	compact = min(solutions, key=lambda s: s.nLookups * s.fullCost * s.fullCost)
	print("Compact solution:", compact)
	print_solution(compact)


def main(args=sys.argv):

	print("General_Category:")
	f, default = ucd.category, 'Cn'
	gc_data = [f(chr(u)) for u in range(0x110000)]
	solve(gc_data, default)
	print()

	print("Canonical_Combining_Class:")
	f, default = ucd.combining, 0
	ccc_data = [f(chr(u)) for u in range(0x110000)]
	solve(ccc_data, default)
	print()

	print("General_Category and Canonical_Combining_Class combined:")
	f, default = ucd.combining, 0
	gc_ccc_data = [gc+str(ccc) for gc,ccc in zip(gc_data, ccc_data)]
	solve(gc_ccc_data, default)
	print()

	print("Mirrored:")
	f, default = ucd.mirrored, 0
	mirrored_data = [f(chr(u)) for u in range(0x110000)]
	solve(mirrored_data, default)
	print()

	print("GC of all mirrored characters:")
	mirrored_gcs = [gc for m,gc in zip(mirrored_data, gc_data) if m]
	print(set(mirrored_gcs))
	print()

	return 0

if __name__ == "__main__":
	sys.exit(main())
