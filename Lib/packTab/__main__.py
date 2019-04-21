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

def main(args=sys.argv):
	f = ucd.category
	data = [f(chr(u)) for u in range(0x110000)]
	solutions = pack_table(data, None, 'Cn').solutions
	for s in solutions:
		print(s.nLookups * s.fullCost, s.fullCost, s)
	return 0

if __name__ == "__main__":
	sys.exit(main())
