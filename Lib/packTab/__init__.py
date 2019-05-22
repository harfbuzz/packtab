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
#
# TODO:
# - Reduce code duplication between Inner/Outer genCode().
# - Handle empty data array.
# - Bake in width multiplier into array data if doing so doesn't enlarge
#   data type.  Again, that would save ops.
# - If an array is not larger than 64 bits, inline it in code directly
#   as one integer.
# - Currently we only cull array of defaults at the end.  Do it at
#   beginning as well, and adjust split code to find optimum shift.
# - Byte reuse!  Much bigger work item.

"""
Pack a static table of integers into compact lookup tables to save space.
"""

from __future__ import print_function, division, absolute_import
import sys
import collections
from math import ceil
from itertools import count
from functools import partial
try:
    from math import log2
except ImportError:
    from math import log
    from functools import partial
    log2 = lambda x: log(x, 2)

if sys.version_info[0] < 3:
    _float_ceil = ceil
    ceil = lambda x: int(_float_ceil(x))


__all__ = ['Code', 'pack_table', 'pick_solution']

__version__ = "0.1.0"


class AutoMapping(collections.defaultdict):
    _next = 0
    def __missing__(self, key):
        assert not isinstance(key, int)
        v = self._next
        self._next = self._next + 1
        self[key] = v
        self[v] = key
        return v

def binaryBitsFor(minV, maxV):
    """Returns smallest power-of-two number of bits needed to represent n
    different values.

    >>> binaryBitsFor(0, 0)
    0
    >>> binaryBitsFor(0, 1)
    1
    >>> binaryBitsFor(0, 6)
    4
    >>> binaryBitsFor(0, 14)
    4
    >>> binaryBitsFor(0, 15)
    4
    >>> binaryBitsFor(0, 16)
    8
    >>> binaryBitsFor(0, 100)
    8
    """

    assert minV <= maxV

    if 0 <= minV and maxV <= 0: return 0
    if 0 <= minV and maxV <= 1: return 1
    if 0 <= minV and maxV <= 3: return 2
    if 0 <= minV and maxV <= 15: return 4

    if 0 <= minV and maxV <= 255: return 8
    if -128 <= minV and maxV <= 127: return 8

    if 0 <= minV and maxV <= 65535: return 16
    if -32768 <= minV and maxV <= 32767: return 16

    if 0 <= minV and maxV <= 4294967295: return 32
    if -2147483648 <= minV and maxV <= 2147483647: return 32

    if 0 <= minV and maxV <= 18446744073709551615: return 64
    if -9223372036854775808 <= minV and maxV <= 9223372036854775807: return 64

    assert False


def print_array(typ, name, values,
                print=print,
                linkage='static const'):

    if linkage: linkage += ' '

    # Make sure we can read multiple times from values:
    assert len(values) == len(values)

    print('%s%s' % (linkage, typ))
    print('%s[%s] =' % (name, len(values)))
    print('{')
    w = max(len(str(v)) for v in values)
    n = 1 << int(log2(78 / (w + 1)))
    if (w + 2) * n <= 78:
        w += 1
    for i in range(0, len(values), n):
        line = values[i:i+n]
        print('  ' + ''.join('%*s,' % (w, v) for v in line))
    print('};')

def print_function(ret, name, args, body,
                   print=print,
                   linkage='static const'):

    if linkage: linkage += ' '

    args = ', '.join(' '.join(p) for p in args)

    print('%s%s' % (linkage, ret))
    print('%s (%s)' % (name, args))
    print('{')
    print('  return %s;' % body)
    print('}')


class Code:
    def __init__(self, namespace=''):
        self.namespace = namespace
        self.functions = collections.OrderedDict()
        self.arrays = collections.OrderedDict()

    def nameFor(self, name):
        return '%s_%s' % (self.namespace, name)

    def addFunction(self, linkage, retType, name, args, body):
        name = self.nameFor(name)
        key = (linkage, retType, name, args)
        if key in self.functions:
            assert self.functions[key] == body
        else:
            self.functions[key] = body
        return name

    def addArray(self, typ, name, values=[]):
        name = self.nameFor(name)
        key = (typ, name)
        array = self.arrays.setdefault(key, [])
        start = len(array)
        array.extend(values)
        return name, array, start

    def print_c(self,
                file=sys.stdout,
                linkage='',
                indent=0):
        if isinstance(indent, int): indent *= ' '
        printn = partial(print, file=file, sep='')
        println = partial(printn, indent)

        for (typ, name), values in self.arrays.items():
            print_array(typ, name, values, println)

        if self.arrays and self.functions:
            printn()

        for (link, ret, name, args), body in self.functions.items():
            link = linkage if link is None else link
            print_function(ret, name, args, body, println, linkage=link)

    def print_h(self,
                file=sys.stdout,
                linkage='',
                indent=0):
        if linkage: linkage += ' '
        if isinstance(indent, int): indent *= ' '
        printn = partial(print, file=file, sep='')
        println = partial(printn, indent)

        for (link, ret, name, args), body in self.functions.items():
            link = linkage if link is None else link+' '
            args = ', '.join(' '.join(p) for p in args)
            println('%s%s %s (%s);' % (linkage, ret, name, args))

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

    assert minV <= maxV

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

    def genCode(self, code, name=None, var='u'):
        inputVar = var
        if name: var = 'u'
        expr = var

        typ = typeFor(0, self.layer.maxV)
        retType = fastType(typ)
        unitBits = self.layer.unitBits
        if not unitBits:
            expr = self.layer.data[0]
            return (retType, expr)

        arrName, array, start = code.addArray(typ, typeAbbr(typ))

        shift = self.bits
        width = 1 << shift
        mask = width - 1

        if self.next:
            (_,expr) = self.next.genCode(code, None, "%s>>%d" % (var, shift))

        start = str(start)+'+' if start else ''
        if expr == '0' or width == 0:
            index0 = ''
        elif width == 1:
            index0 = str(expr)
        else:
            index0 = '%d*(%s)' % (width, expr)
        index1 = '((%s)&%d)' % (var, mask) if mask else ''
        index = index0 + ('+' if index0 and index1 else '') + index1
        if unitBits >= 8:
            if start:
                index = '(%s)' % index
            expr = '%s[%s%s]' % (arrName, start, index)
        else:
            shiftBits = int(round(log2(8 // unitBits)))
            mask1 = (8 // unitBits) - 1
            mask2 = (1 << unitBits) - 1
            funcBody = '(a[i>>%s]>>(i&%s))&%s' % (shiftBits, mask1, mask2)
            funcName = code.addFunction ('static inline',
                                         'unsigned',
                                         'b%s' % unitBits,
                                         (('const uint8_t*', 'a'),
                                          ('unsigned',       'i')),
                                         funcBody)
            expr = '%s(%s%s,%s)' % (funcName, start, arrName, index)

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
            for d in range(layer.maxV + 1):
                _expand(d, layers, len(layers) - 1, data)

        data = _combine(data, self.layer.unitBits)
        array.extend(data)

        if name:
            funcName = code.addFunction (None,
                                         retType,
                                         name,
                                         (('unsigned', 'u'),),
                                         expr)
            expr = '%s(%s)' % (funcName, inputVar)

        return (retType, expr)

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

    def __init__(self, data):
        self.data = data
        self.next = None
        self.solutions = []

class InnerLayer(Layer):

    """
    A layer that can reproduce @data passed to its constructor, by
    using multiple lookup tables that split the domain by powers
    of two.
    """

    def __init__(self, data):
        Layer.__init__(self, data)

        self.maxV = max(data)
        self.unitBits = binaryBitsFor(0, self.maxV)
        self.extraOps = subByteAccessOps if self.unitBits < 8 else 0
        self.bytes = ceil(self.unitBits * len(self.data) / 8)

        if self.maxV == 0:
            return

        self.split()

        solution = InnerSolution(self,
                                 None,
                                 1 if self.maxV > 0 else 0,
                                 self.extraOps,
                                 self.bytes)
        self.solutions.append(solution)

        bits = 1
        layer = self.next
        while layer is not None:

            extraCost = ceil((layer.maxV + 1) * (1<<bits) * self.unitBits / 8)

            for s in layer.solutions:
                nLookups = s.nLookups + 1
                nExtraOps = s.nExtraOps + self.extraOps
                cost = s.cost + extraCost
                solution = InnerSolution(self, s, nLookups, nExtraOps, cost, bits)
                self.solutions.append(solution)

            layer = layer.next
            bits += 1

        self.prune_solutions()

    def split(self):
        if len(self.data) & 1:
            self.data.append(0)

        mapping = self.mapping = AutoMapping()
        data2 = _combine2(self.data, lambda a,b: mapping[(a,b)])

        self.next = InnerLayer(data2)

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

    def genCode(self, code, name=None, var='u'):
        inputVar = var
        if name: var = 'u'
        expr = var

        typ = typeFor(self.layer.minV, self.layer.maxV)
        retType = fastType(typ)
        unitBits = self.layer.unitBits
        if not unitBits:
            assert False # Audit this branch
            expr = self.layer.data[0]
            return (retType, expr)

        if self.next:
            (_,expr) = self.next.genCode(code, None, var)

        if self.layer.mult != 1:
            expr = '%d*%s' % (self.layer.mult, expr)
        if self.layer.bias != 0:
            expr = '%d+%s' % (self.layer.bias, expr)

        expr = '%s<%d?%s:%s' % (var,
                                len(self.layer.data),
                                expr,
                                self.layer.default) # TODO Map default?

        if name:
            funcName = code.addFunction (None,
                                         retType,
                                         name,
                                         (('unsigned', 'u'),),
                                         expr)
            expr = '%s(%s)' % (funcName, inputVar)

        return (retType, expr)


def gcd(lst):
    """
    >>> gcd([])
    1
    >>> gcd([48])
    48
    >>> gcd([48, 60])
    12
    >>> gcd([48, 60, 6])
    6
    >>> gcd([48, 61, 6])
    1
    """
    it = iter(lst)
    try:
        x = next(it)
    except StopIteration:
        return 1
    for y in it:
        while y:
            x, y = y, x%y
        if x == 1:
            return 1
    return x

class OuterLayer(Layer):

    """
    A layer that can reproduce @data passed to its constructor, by
    simple arithmetic tricks to reduce its size.
    """

    def __init__(self, data, default):
        data = list(data)
        while data[-1] == default:
            data.pop()
        Layer.__init__(self, data)
        self.default = default

        self.minV, self.maxV = min(data), max(data)

        bias = 0
        mult = 1
        unitBits = binaryBitsFor(self.minV, self.maxV)

        b = self.minV
        if unitBits > binaryBitsFor(0, self.maxV - b):
            unitBits = binaryBitsFor(0, self.maxV - b)
            bias = b

        m = gcd(data)
        if unitBits > binaryBitsFor(self.minV // m, self.maxV // m):
            unitBits = binaryBitsFor(self.minV // m, self.maxV // m)
            bias = 0
            mult = m

        if b:
            m = gcd(d - b for d in data)
            if unitBits > binaryBitsFor(0, (self.maxV - b) // m):
                unitBits = binaryBitsFor(0, (self.maxV - b) // m)
                bias = b
                mult = m

        self.unitBits = unitBits
        self.extraOps = subByteAccessOps if self.unitBits < 8 else 0
        self.bias = bias
        if bias: self.extraOps += 1
        self.mult = mult
        if mult: self.extraOps += 1
        data = [(d - bias) // mult for d in self.data]
        default = (self.default - bias) // mult

        self.bytes = ceil(self.unitBits * len(self.data) / 8)
        self.next = InnerLayer(data)

        extraCost = 0

        layer = self.next
        for s in layer.solutions:
            nLookups = s.nLookups
            nExtraOps = s.nExtraOps + self.extraOps
            cost = s.cost + extraCost
            solution = OuterSolution(self, s, nLookups, nExtraOps, cost)
            self.solutions.append(solution)


# Public API

def pack_table(data, default=0, compression=1, mapping=None):
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

    TODO: Mapping does not work currently.
    """

    # Set up mapping.  See docstring.
    if mapping is not None:
        #assert (all(isinstance(k, int) and not isinstance(v, int) for k,v in mapping.items()) or
        #        all(not isinstance(k, int) and isinstance(v, int) for k,v in mapping.items()))
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
        all(not isinstance(v, int) for v in data)), data
    if not isinstance(data[0], int):
        data = [mapping[v] for v in data]
    if not isinstance(default, int):
        default = mapping[default]

    solutions = OuterLayer(data, default).solutions

    if compression is None:
        solutions.sort(key=lambda s: -s.fullCost)
        return solutions

    return pick_solution(solutions, compression)

def pick_solution(solutions, compression=1):
    return min(solutions, key=lambda s: s.nLookups + compression*log2(s.fullCost))


if __name__ == "__main__":
    import doctest
    sys.exit(doctest.testmod().failed)
