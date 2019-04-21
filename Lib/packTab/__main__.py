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
import sys
import unicodedata as ucd

try:
	chr = unichr
	try:
		unichr(0x10000)
	except ValueError:
		raise Exception("Narrow Python build.  Use Python3 instead.")
except NameError:
	pass

def solve(data, default):
	print("Unique values:", len(set(data)))
	solutions = pack_table(data, None, default).solutions
	print("All dominant solutions:")
	for s in solutions:
		print(s)
	optimal = min(solutions, key=lambda s: s.nLookups * s.nLookups * s.fullCost)
	compact = min(solutions, key=lambda s: s.nLookups * s.fullCost * s.fullCost)
	print ("Optimal solution:", optimal)
	print ("Compact solution:", compact)

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
