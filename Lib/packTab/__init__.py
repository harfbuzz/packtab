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


__all__ = ['pack_table', 'pick_solution']


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
    return 1 << int(ceil(log2(log2(n))))

bytesPerOp = 4
lookupOps = 4
subByteAccessOps = 4

class Solution:
    def __init__(self, layer, next, nLookups, nExtraOps, cost):
        self.layer = layer
        self.next = next
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

    if 0 <= minV and maxV <= 18446744073709551615: return 'uint64_t'
    if -9223372036854775808 <= minV and maxV <= 9223372036854775807: return 'int64_t'

    assert False

def typeWidth(typ):
    """
    >>> typeWidth('int8_t')
    8
    >>> typeWidth('uint32_t')
    32
    """
    return int(typ[typ.index('int')+3:-2])

def typeAbbr(typ):
    """
    >>> typeAbbr('int8_t')
    'i8'
    >>> typeAbbr('uint32_t')
    'u32'
    """
    return typ[0]+str(typeWidth(typ))

def fastType(typ):
    """
    >>> fastType('int8_t')
    'int_fast8_t'
    >>> fastType('uint32_t')
    'uint_fast32_t'
    """
    return typ.replace('int', 'int_fast')


class InnerSolution(Solution):

    def __init__(self, layer, next, nLookups, nExtraOps, cost, bits=0):
        Solution.__init__(self, layer, next, nLookups, nExtraOps, cost)
        self.bits = bits

    def __repr__(self):
        return "%s%s" % (self.__class__.__name__,
               (self.nLookups, self.nExtraOps, self.cost, self.bits))

    def genCode(self, prefix='', var='u', functions=None, arrays=None):

        if functions is None:
            functions = collections.OrderedDict()
        if arrays is None:
            arrays = collections.OrderedDict()
        expr = var

        typ = typeFor(self.layer.minV, self.layer.maxV)
        arrName = prefix+'_'+typeAbbr(typ)
        unitBits = self.layer.unitBits
        if not unitBits:
            expr = self.layer.data[0]
            return functions, arrays, (fastType(typ), expr)

        array = arrays.setdefault((typ, arrName), [])
        start = len(array)

        shift = self.bits
        width = 1 << shift
        mask = width - 1

        if self.next:
            functions, arrays, (_,expr) = self.next.genCode(prefix,
                                                            "%s>>%d" % (var, shift),
                                                            functions, arrays)

        start = str(start)+'+' if start else ''
        if expr == '0' or width == 0:
            index0 = ''
        elif width == 1:
            index0 = str(expr)
        else:
            index0 = '%d*(%s)' % (width, expr)
        index1 = '(%s)&%d' % (var, mask) if mask else ''
        index = index0 + ('+' if index0 and index1 else '') + index1
        if unitBits >= 8:
            expr = '%s[%s%s]' % (arrName, start, index)
        else:
            expr = '%s_b%s(%s%s,%s)' % (prefix, unitBits, arrName, start, index)
            shiftBits = int(round(log2(8 // unitBits)))
            mask1 = (8 // unitBits) - 1
            mask2 = (1 << unitBits) - 1
            functions[('unsigned', '%s_b%d' % (prefix, unitBits),
                       'const uint8_t *a, unsigned i')] = 'return (a[i>>%s]>>(i&%s))&%s;' % (shiftBits, mask1, mask2)

        layers = []
        layer = self.layer
        bits = self.bits
        while bits:
            layers.append(layer)
            layer = layer.next
            bits -= 1

        data = []
        if not layers:
            mapping = layer.mapping
            # TODO Following is always true.
            if isinstance(layer.data[0], int):
                data.extend(layer.data)
            else:
                data.extend(mapping[d] for d in layer.data)
        else:
            assert layer.minV == 0, layer.minV
            for d in range(layer.maxV + 1):
                _expand(d, layers, len(layers) - 1, data)

        data = _combine(data, self.layer.unitBits)
        array.extend(data)

        return functions, arrays, (fastType(typ), expr)

def _expand(v, stack, i, out):
    if i < 0:
        out.append(v)
        return
    v = stack[i].mapping[v]
    i -= 1
    _expand(v[0], stack, i, out)
    _expand(v[1], stack, i, out)

def _combine(data, bits):
    if bits <= 1: data = _combine2(data, lambda a,b: (b<<1)|a)
    if bits <= 2: data = _combine2(data, lambda a,b: (b<<2)|a)
    if bits <= 4: data = _combine2(data, lambda a,b: (b<<4)|a)
    return data

def _combine2(data, f):
    data2 = []
    it = iter(data)
    for first in it:
        try:
            data2.append(f(first, next(it)))
        except StopIteration:
            data2.append(f(first, 0))
    return data2


class Layer:

    def __init__(self, data, default):
        self.data = data
        self.default = default
        self.next = None
        self.solutions = []

class InnerLayer(Layer):

    """
    A layer that can reproduce @data passed to its constructor, by
    using multiple lookup tables that split the domain by powers
    of two.
    """

    def __init__(self, data, default):
        Layer.__init__(self, data, default)

        self.minV, self.maxV = min(data), max(data)
        self.bandwidth = self.maxV - self.minV + 1
        self.unitBits = binaryBitsFor(self.bandwidth)
        self.extraOps = subByteAccessOps if self.unitBits < 8 else 0
        self.bytes = ceil(self.unitBits * len(self.data) / 8)

        assert self.minV == 0

        if self.bandwidth == 1:
            return

        self.split()

    def split(self):
        if len(self.data) & 1:
            self.data.append(self.default) # TODO Don't modify?

        mapping = self.mapping = AutoMapping()
        default2 = 0#mapping[(self.default, self.default)]
        data2 = _combine2(self.data, lambda a,b: mapping[(a,b)])

        self.next = InnerLayer(data2, default2)

    def solve(self):

        solution = InnerSolution(self,
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

            extraCost = ceil(layer.bandwidth * (1<<bits) * self.unitBits / 8)

            for s in layer.solutions:
                nLookups = s.nLookups + 1
                nExtraOps = s.nExtraOps + self.extraOps
                cost = s.cost + extraCost
                solution = InnerSolution(self, s, nLookups, nExtraOps, cost, bits)
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

class OuterSolution(Solution):

    def __init__(self, layer, next, nLookups, nExtraOps, cost):
        Solution.__init__(self, layer, next, nLookups, nExtraOps, cost)

    def __repr__(self):
        return "%s%s" % (self.__class__.__name__,
               (self.nLookups, self.nExtraOps, self.cost))

    def genCode(self, prefix='', var='u', functions=None, arrays=None):

        if functions is None:
            functions = collections.OrderedDict()
        if arrays is None:
            arrays = collections.OrderedDict()
        expr = var

        typ = typeFor(self.layer.minV, self.layer.maxV)
        arrName = prefix+'_'+typeAbbr(typ)
        unitBits = self.layer.unitBits
        if not unitBits:
            assert False # Audit this branch
            expr = self.layer.data[0]
            return functions, arrays, (fastType(typ), expr)

        if self.next:
            functions, arrays, (_,expr) = self.next.genCode(prefix,
                                                            var,
                                                            functions, arrays)

        if self.layer.bias:
            expr = '%d+%s' % (self.layer.bias, expr)

        return functions, arrays, (fastType(typ), expr)

class OuterLayer(Layer):

    """
    A layer that can reproduce @data passed to its constructor, by
    simple arithmetic tricks to reduce its size.
    """

    def __init__(self, data, default):
        Layer.__init__(self, data, default)

        self.minV, self.maxV = min(data), max(data)
        self.bandwidth = self.maxV - self.minV + 1
        self.unitBits = binaryBitsFor(self.bandwidth)
        self.extraOps = subByteAccessOps if self.unitBits < 8 else 0
        self.bytes = ceil(self.unitBits * len(self.data) / 8)

        bias = self.bias = self.minV
        data = [d - bias for d in self.data]
        if bias:
            self.extraOps += 1
        self.next = InnerLayer(data, self.default)

    def solve(self):

        extraCost = 0

        layer = self.next
        layer.solve()
        for s in layer.solutions:
            nLookups = s.nLookups
            nExtraOps = s.nExtraOps + self.extraOps
            cost = s.cost + extraCost
            solution = OuterSolution(self, s, nLookups, nExtraOps, cost)
            self.solutions.append(solution)

def solve(data, default):

    layer = OuterLayer(data, default)
    layer.solve()
    return layer


# Public API

def pack_table(data, mapping=None, default=0, compression=1):
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

    solutions = solve(data, default).solutions

    if compression is None:
        solutions.sort(key=lambda s: -s.fullCost)
        return solutions

    return pick_solution(solutions, compression)

def pick_solution(solutions, compression=1):
    return min(solutions, key=lambda s: s.nLookups + compression*log2(s.fullCost))


if __name__ == "__main__":
    import doctest
    sys.exit(doctest.testmod().failed)
