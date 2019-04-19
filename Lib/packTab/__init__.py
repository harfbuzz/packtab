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
import array
import collections

class defaultMapping(collections.defaultdict):
	_next = 0
	def __missing__(self, key):
		assert type(key) is str
		v = self._next
		self._next = self._next + 1
		self[key] = v
		self[v] = key
		return v

def pack_table(data, mapping=None, default=0):
	"""

	@data is either a dictionary mapping integer keys to values, of an
	iterable containing values for keys starting at zero.  Values must
	all be integers, or all strings.

	@mapping, if set, should be either a mapping from integer keys to
	string values, or vice versa.  Either way, it's first augmented by its
	own inverse.  After that it's used to map any value in @data that is
	not an integer, to obtain its integer value used for packing size
	considerations.  When generating output table, integer values are tried
	mapped through mapping to obtain string mnemonic to write out.  If such
	mapping does not exist, the integer value will be written out.

	If mapping is not set, it will be automatically populated to assign
	increasing integers starting from zero, to every new string key in
	@data.  This internal mapping only affects value size assumptions, but
	will not otherwise be visible in the output.

	@default is value to be used for keys not specified in @data.  Defaults
	to zero.  If data values are strings and @mapping is not provided, then
	@default must be specified, or bad things might happen.

	"""

	# Set up mapping.  See docstring.
	if mapping is not None:
		assert (all(type(k) is int and type(v) is str for k,v in mapping.items()) or
			all(type(k) is str and type(v) is int for k,v in mapping.items()))
		mapping2 = mapping.copy()
		for k,v in mapping.items():
			mapping2[v] = k
		mapping = mapping2
		del mapping2
	else:
		mapping = defaultMapping()

	# Set up data as a list.  TODO: Skip list intermediate.
	if isinstance(data, dict):
		assert(all(type(k) is int and type(v) in (int, str) for k,v in data.items()))
		minK = min(dict.keys())
		maxK = max(dict.keys())
		assert minK >= 0
		data2 = [default] * (maxK + 1)
		for k,v in data.items():
			data2[k] = v
		data = data2
		del data2

	# Set up data as an array.
	assert (all(type(v) is int for v in data) or
		all(type(v) is str for v in data))
	if type(data[0]) is str:
		data = [mapping[v] for v in data]


