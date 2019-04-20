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
import collections
from math import ceil
try:
	from math import log2
except ImportError:
	from math import log
	from functools import partial
	log2 = partial(log, base=2)

class AutoMapping(collections.defaultdict):
	_next = 0
	def __missing__(self, key):
		assert type(key) is not int
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
		mapping = AutoMapping()

	# Set up data as a list.
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

	# Convert all to integers
	assert (all(type(v) is int for v in data) or
		all(type(v) is str for v in data))
	if type(data[0]) is str:
		data = [mapping[v] for v in data]
	if type(default) is str:
		default = mapping[default]

	return solve(data, default)

def binaryBitsFor(n):
	"""Returns smalles power-of-two number of bits needed to represent n
	different values.

	>>> binaryBitsFor(1)
	0
	>>> binaryBitsFor(2)
	1
	>>> binaryBitsFor(7)
	4
	>>> binaryBitsFor(15)
	4
	>>> binaryBitsFor(16)
	4
	>>> binaryBitsFor(17)
	8
	>>> binaryBitsFor(100)
	8
	"""
	if n is 1: return 0
	return 1 << ceil(log2(log2(n)))

class BinarySolution:
	def __init__(self, nLookups, nExtraOps, cost, mult=0):
		self.nLookups = nLookups
		self.nExtraOps = nExtraOps
		self.cost = cost
		self.mult = mult

		self.key = (nLookups, nExtraOps)

	def __repr__(self):
		return "BinarySolution(%d,%d,%d,%d)" % \
			(self.nLookups, self.nExtraOps, self.cost, self.mult)

class BinaryLayer:

	"""
	A layer that can reproduce @data passed to its constructor, by
	using multiple lookup tables that split the domain by powers
	of two.
	"""

	def __init__(self, data, default):
		self.data = data
		self.default = default
		self.next = None
		self.solutions = {}

		self.minV, self.maxV = min(data), max(data)
		self.bandwidth = self.maxV - self.minV + 1
		self.unitBits = binaryBitsFor(self.bandwidth)
		self.extraOps = 1 if self.unitBits < 8 else 0
		self.bytes = ceil(self.unitBits * len(self.data) / 8)

		if len(data) is 1 or self.bandwidth is 1:
			return

		self.split()

	def split(self):
		if len(self.data) & 1:
			self.data.append(self.default) # TODO Don't modify?

		mapping = self.mapping = AutoMapping()
		default2 = mapping[(self.default, self.default)]
		data2 = []
		it = iter(self.data)
		for first in it: data2.append(mapping[(first, next(it))])

		self.next = BinaryLayer(data2, default2)

	def solve(self):

		solution = BinarySolution(1 if self.bandwidth > 1 else 0,
					  self.extraOps,
					  self.bytes)
		self.solutions[solution.key] = solution
		if self.next is None:
			return
		self.next.solve()

		mult = 2
		layer = self.next
		while layer is not None:

			extraCost = ceil(layer.bandwidth * mult * self.unitBits / 8)

			for s in layer.solutions.values():
				nLookups = s.nLookups + 1
				nExtraOps = s.nExtraOps + self.extraOps
				cost = s.cost + extraCost
				solution = BinarySolution(nLookups, nExtraOps, cost, mult)
				if (solution.key not in self.solutions or
				    solution.cost < self.solutions[solution.key].cost):
					self.solutions[solution.key] = solution

			layer = layer.next
			mult <<= 1

def solve(data, default):

	layer = BinaryLayer(data, default)
	layer.solve()
	return layer.solutions


if __name__ == "__main__":
	import sys, doctest
	sys.exit(doctest.testmod().failed)
