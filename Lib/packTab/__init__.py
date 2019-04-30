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

# I first wrote something like this back in 2001 when I needed it in FriBidi:
#
#   https://github.com/fribidi/fribidi/blob/master/gen.tab/packtab.c
#
# In 2019 I wanted to use that to produce more compact Unicode data tables
# for HarfBuzz, but for convenience I wanted to use it from Python.  While
# I considered wrapping the C code in a module, it occurred to me that I
# can rewrite it in pure Python in a much cleaner way.  That code remains
# a stain on my resume in terms of readability (or lack thereof!). :D
#
# This Python version builds on the same ideas, but is different from the
# C version in two major ways:
#
# 1. Whereas the C version uses backtracking to find best split opportunities,
#    I found that the same can be achieved using dynamic-programming.  So the
#    Python version implements the DP approach, which is much faster.
#
# 2. The C version does not try packing multiple items into a single byte.
#    The Python version does.  Ie. if items fit, they might get packed into
#    1, 2, or 4 bits per item.
#
# There's also a bunch of other optimizations, which make (eventually, when
# complete) the Python version more generic and usable for a wider variety
# of data tables.

"""
Pack a static table of integers into compact lookup tables to save space.
"""

from __future__ import print_function, division, absolute_import
import sys
import collections
from math import ceil
from itertools import count
try:
	from math import log2
except ImportError:
	from math import log
	from functools import partial
	log2 = lambda x: log(x, 2)

if sys.version_info[0] < 3:
	_float_ceil = ceil
	ceil = lambda x: int(_float_ceil(x))


__all__ = ['pack_table']


class AutoMapping(collections.defaultdict):
	_next = 0
	def __missing__(self, key):
		assert not isinstance(key, int)
		v = self._next
		self._next = self._next + 1
		self[key] = v
		self[v] = key
		return v

def binaryBitsFor(n):
	"""Returns smallest power-of-two number of bits needed to represent n
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
	if n == 1: return 0
	return 1 << ceil(log2(log2(n)))

bytesPerOp = 4
lookupOps = 4
subByteAccessOps = 4

class Solution:
	def __init__(self, layer, nxt, nLookups, nExtraOps, cost):
		self.layer = layer
		self.nxt = nxt
		self.nLookups = nLookups
		self.nExtraOps = nExtraOps
		self.cost = cost

	@property
	def fullCost(self):
		return self.cost + (self.nLookups * lookupOps + self.nExtraOps) * bytesPerOp

	def __repr__(self):
		return "%s%s" % (self.__class__.__name__,
			(self.nLookups, self.nExtraOps, self.cost))

def typeFor(minV, maxV):

	if 0 <= minV and maxV <= 255: return 'uint8_t'
	if -128 <= minV and maxV <= 127: return 'int8_t'

	if 0 <= minV and maxV <= 65535: return 'uint16_t'
	if -32768 <= minV and maxV <= 32767: return 'int16_t'

	if 0 <= minV and maxV <= 4294967295: return 'uint32_t'
	if -2147483648 <= minV and maxV <= 2147483647: return 'int32_t'

	if 0 <= minV: return 'uint64_t'
	return 'int64_t'

class BinarySolution(Solution):

	def __init__(self, layer, nxt, nLookups, nExtraOps, cost, bits=0):
		Solution.__init__(self, layer, nxt, nLookups, nExtraOps, cost)
		self.bits = bits

	def __repr__(self):
		return "%s%s" % (self.__class__.__name__,
			(self.nLookups, self.nExtraOps, self.cost, self.bits))

	def genCode(self, var, prefix='', functions=None, arrays=None):

		typ = typeFor(self.layer.minV, self.layer.maxV)
		name = prefix+'_'+typ[0]+typ[typ.index('int')+3:-2]
		arr = arrays.setdefault((typ, name), [])
		off = len(arr)

		if functions is None:
			functions = collections.OrderedDict()
		if arrays is None:
			arrays = collections.OrderedDict()
		expr = var

		if self.nxt:
			functions, arrays, nxtExpr = self.nxt.genCode("var/%s"%(1<<self.bits), prefix, functions, arrays)
			expr = '%s[%d+%s]+%s'%(name, off, nxtExpr, "(("+var+"/12)&3)")

		if self.layer.unitBits == 1:
			functions[('unsigned', prefix+'_b1', 'const uint8_t *a, unsigned i')] = 'return (a[i>>3] >> (i&7)) & 1;'
		elif self.layer.unitBits == 2:
			functions[('unsigned', prefix+'_b2', 'const uint8_t *a, unsigned i')] = 'return (a[i>>2] >> (i&3)) & 3;'
		elif self.layer.unitBits == 4:
			functions[('unsigned', prefix+'_b4', 'const uint8_t *a, unsigned i')] = 'return (a[i>>1] >> (i&1)) & 7;'

		for i in range(96):
			arr.append(i)

		return functions, arrays, expr

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
		self.solutions = []

		self.minV, self.maxV = min(data), max(data)
		self.bandwidth = self.maxV - self.minV + 1
		self.unitBits = binaryBitsFor(self.bandwidth)
		self.extraOps = subByteAccessOps if self.unitBits < 8 else 0
		self.bytes = ceil(self.unitBits * len(self.data) / 8)

		if len(data) == 1 or self.bandwidth == 1:
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

		solution = BinarySolution(self,
					  None,
					  1 if self.bandwidth > 1 else 0,
					  self.extraOps,
					  self.bytes)
		self.solutions.append(solution)
		if self.next is None:
			return
		self.next.solve()

		bits = 1
		layer = self.next
		while layer is not None:

			extraCost = ceil(layer.bandwidth * (2<<bits) * self.unitBits / 8)

			for s in layer.solutions:
				nLookups = s.nLookups + 1
				nExtraOps = s.nExtraOps + self.extraOps
				cost = s.cost + extraCost
				solution = BinarySolution(self, s, nLookups, nExtraOps, cost, bits)
				self.solutions.append(solution)

			layer = layer.next
			bits += 1

		self.prune_solutions()

	def prune_solutions(self):
		"""Remove dominated solutions."""

		# Doing it the slowest, O(N^2), way for now.
		sols = self.solutions
		for a in sols:
			if a.cost == None: continue
			for b in sols:
				if a is b: continue
				if b.cost == None: continue

				# Rules of dominance: a being not worse than b
				if a.nLookups <= b.nLookups and a.fullCost <= b.fullCost:
					b.cost = None
					continue

		self.solutions = [s for s in self.solutions if s.cost is not None]
		self.solutions.sort(key=lambda s: s.nLookups)

def solve(data, default):

	layer = BinaryLayer(data, default)
	layer.solve()
	return layer


# Public API
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
		assert (all(isinstance(k, int) and not isinstance(v, int) for k,v in mapping.items()) or
			all(not isinstance(k, int) and isinstance(v, int) for k,v in mapping.items()))
		mapping2 = mapping.copy()
		for k,v in mapping.items():
			mapping2[v] = k
		mapping = mapping2
		del mapping2
	else:
		mapping = AutoMapping()

	# Set up data as a list.
	if isinstance(data, dict):
		assert(all(isinstance(k, int) for k,v in data.items()))
		minK = min(data.keys())
		maxK = max(data.keys())
		assert minK >= 0
		data2 = [default] * (maxK + 1)
		for k,v in data.items():
			data2[k] = v
		data = data2
		del data2

	# Convert all to integers
	assert (all(isinstance(v, int) for v in data) or
		all(not isinstance(v, int) for v in data))
	if not isinstance(data[0], int):
		data = [mapping[v] for v in data]
	if not isinstance(default, int):
		default = mapping[default]

	return solve(data, default)


if __name__ == "__main__":
	import doctest
	sys.exit(doctest.testmod().failed)
