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


__all__ = ["Code", "pack_table", "pick_solution", "languages"]

__version__ = "0.3.0"


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

    if type(minV) != int or type(maxV) != int:
        return 8

    assert minV <= maxV

    if 0 <= minV and maxV <= 0:
        return 0
    if 0 <= minV and maxV <= 1:
        return 1
    if 0 <= minV and maxV <= 3:
        return 2
    if 0 <= minV and maxV <= 15:
        return 4

    if 0 <= minV and maxV <= 255:
        return 8
    if -128 <= minV and maxV <= 127:
        return 8

    if 0 <= minV and maxV <= 65535:
        return 16
    if -32768 <= minV and maxV <= 32767:
        return 16

    if 0 <= minV and maxV <= 4294967295:
        return 32
    if -2147483648 <= minV and maxV <= 2147483647:
        return 32

    if 0 <= minV and maxV <= 18446744073709551615:
        return 64
    if -9223372036854775808 <= minV and maxV <= 9223372036854775807:
        return 64

    assert False


class Language:
    def print_array(self, name, array, *, print=print, private=True):
        linkage = self.private_array_linkage if private else self.public_array_linkage
        decl = self.declare_array(linkage, array.typ, name, len(array.values))
        print(decl, "=")
        print(self.array_start)
        w = max((len(str(v)) for v in array.values), default=1)
        n = 1 << int(round(log2(78 / (w + 1))))
        if (w + 2) * n <= 78:
            w += 1
        for i in range(0, len(array.values), n):
            line = array.values[i : i + n]
            print("  " + "".join("%*s," % (w, v) for v in line))
        print(self.array_end)

    def print_function(self, name, function, *, print=print):
        linkage = (
            self.private_function_linkage
            if function.private
            else self.public_function_linkage
        )
        decl = self.declare_function(linkage, function.retType, name, function.args)
        print(decl)
        print(self.function_start)
        print("  %s" % self.return_stmt(function.body))
        print(self.function_end)

    unsafe_array_access = False

    def array_index(self, name, index):
        return "%s[%s]" % (name, index)

    def usize_literal(self, value):
        if value == '':
            return ''
        return "%s%s" % (value, self.usize_suffix)


class LanguageC(Language):
    name = "c"
    private_array_linkage = "static const"
    public_array_linkage = "extern const"
    private_function_linkage = "static inline"
    public_function_linkage = "extern inline"
    array_start = "{"
    array_end = "};"
    function_start = "{"
    function_end = "}"
    u8 = "uint8_t"
    usize = "unsigned"
    usize_suffix = "u"

    def print_preamble(self, *, print=print):
        print("#include <stdint.h>")
        print()

    def cast(self, typ, expr):
        return "(%s)(%s)" % (typ, expr)

    def borrow(self, name):
        return "%s" % name

    def slice(self, name, start):
        return "%s+%s" % (name, start)

    def tertiary(self, cond, trueExpr, falseExpr):
        return "%s ? %s : %s" % (cond, trueExpr, falseExpr)

    def declare_array(self, linkage, typ, name, size):
        if linkage:
            linkage += " "
        return "%s%s %s[%d]" % (linkage, typ, name, size)

    def declare_function(self, linkage, retType, name, args):
        if linkage:
            linkage += " "
        args = [(t if t[-1] != "*" else "const %s" % t, n) for t, n in args]
        args = ", ".join(" ".join(p) for p in args)
        return "%s%s %s (%s)" % (linkage, retType, name, args)

    def type_name(self, typ):
        assert typ[0] in "iu"
        signed = "" if typ[0] == "i" else "u"
        size = typeWidth(typ)
        return "%sint%s_t" % (signed, size)

    def type_for(self, minV, maxV):
        assert minV <= maxV

        if type(minV) != int or type(maxV) != int:
            return "uint8_t"

        if 0 <= minV and maxV <= 255:
            return "uint8_t"
        if -128 <= minV and maxV <= 127:
            return "int8_t"

        if 0 <= minV and maxV <= 65535:
            return "uint16_t"
        if -32768 <= minV and maxV <= 32767:
            return "int16_t"

        if 0 <= minV and maxV <= 4294967295:
            return "uint32_t"
        if -2147483648 <= minV and maxV <= 2147483647:
            return "int32_t"

        if 0 <= minV and maxV <= 18446744073709551615:
            return "uint64_t"
        if -9223372036854775808 <= minV and maxV <= 9223372036854775807:
            return "int64_t"

        assert False

    def as_usize(self, expr):
        return expr

    def return_stmt(self, expr):
        return "return %s;" % expr

class LanguageRust(Language):
    name = "rust"
    private_array_linkage = "static"
    public_array_linkage = "pub(crate) static"
    private_function_linkage = ""
    public_function_linkage = "pub(crate)"
    array_start = "["
    array_end = "];"
    function_start = "{"
    function_end = "}"
    u8 = "u8"
    usize = "usize"
    usize_suffix = "usize"

    def print_preamble(self, *, print=print):
        pass

    def cast(self, typ, expr):
        return "(%s) as %s" % (expr, typ)

    def borrow(self, name):
        return "&%s" % name

    def slice(self, name, start):
        return "&%s[%s..]" % (name, start)

    def tertiary(self, cond, trueExpr, falseExpr):
        return "if %s { %s } else { %s }" % (cond, trueExpr, falseExpr)

    def declare_array(self, linkage, typ, name, size):
        if linkage:
            linkage += " "
        typ = "%s" % typ
        return "%s%s: [%s; %d]" % (linkage, name, typ, size)

    def declare_function(self, linkage, retType, name, args):
        if linkage:
            linkage += " "
        retType = "%s" % retType
        args = [(t if t[-1] != "*" else "&[%s]" % t[:-1], n) for t, n in args]
        args = ", ".join("%s: %s" % (n, t) for t, n in args)
        return "%sfn %s (%s) -> %s" % (linkage, name, args, retType)

    def type_name(self, typ):
        assert typ[0] in "iu"
        signed = typ[0]
        size = typeWidth(typ)
        return "%s%s" % (signed, size)

    def type_for(self, minV, maxV):
        assert minV <= maxV

        if type(minV) != int or type(maxV) != int:
            return "u8"

        if 0 <= minV and maxV <= 255:
            return "u8"
        if -128 <= minV and maxV <= 127:
            return "i8"

        if 0 <= minV and maxV <= 65535:
            return "u16"
        if -32768 <= minV and maxV <= 32767:
            return "i16"

        if 0 <= minV and maxV <= 4294967295:
            return "u32"
        if -2147483648 <= minV and maxV <= 2147483647:
            return "i32"

        if 0 <= minV and maxV <= 18446744073709551615:
            return "u64"
        if -9223372036854775808 <= minV and maxV <= 9223372036854775807:
            return "i64"

        assert False

    def as_usize(self, expr):
        if not expr:
            return ''
        try:
            int(expr)
            return "%susize" % expr
        except ValueError:
            # Assume expr is a variable or expression that evaluates to an integer.
            # Rust requires explicit casting to usize.
            if expr.startswith('(') and expr.endswith(')'):
                return "%s as usize" % expr
            else:
                return "(%s) as usize" % expr

    def array_index(self, name, index):
        if self.unsafe_array_access:
            return "unsafe { *(%s.get_unchecked(%s)) }" % (name, index)
        return "%s[%s]" % (name, index)

    def return_stmt(self, expr):
        return expr

languages = {
    "c": LanguageC(),
    "rust": LanguageRust(),
}


class Array:
    def __init__(self, typ):
        self.typ = typ
        self.values = []

    def extend(self, values):
        start = len(self.values)
        self.values.extend(values)
        return start


class Function:
    def __init__(self, retType, args, body, *, private=True):
        self.retType = retType
        self.args = args
        self.body = body
        self.private = private


class Code:
    def __init__(self, namespace=""):
        self.namespace = namespace
        self.functions = collections.OrderedDict()
        self.arrays = collections.OrderedDict()

    def nameFor(self, name):
        return "%s_%s" % (self.namespace, name)

    def addFunction(self, retType, name, args, body, *, private=True):
        name = self.nameFor(name)
        if name in self.functions:
            assert self.functions[name].retType == retType
            assert self.functions[name].args == args
            assert self.functions[name].body == body
            assert self.functions[name].private == private
        else:
            self.functions[name] = Function(retType, args, body, private=private)
        return name

    def addArray(self, typ, name, values):
        name = self.nameFor(name)
        array = self.arrays.get(name)
        if array is None:
            array = self.arrays[name] = Array(typ)
        start = array.extend(values)
        return name, start

    def print_code(self, *, file=sys.stdout, private=True, indent=0, language="c"):
        if isinstance(indent, int):
            indent *= " "
        printn = partial(print, file=file, sep="")
        println = partial(printn, indent)

        if isinstance(language, str):
            language = languages[language]

        language.print_preamble(print=println)

        for name, array in self.arrays.items():
            language.print_array(name, array, print=println, private=private)

        if self.arrays and self.functions:
            printn()

        for name, function in self.functions.items():
            language.print_function(name, function, print=println)

    def print_c(self, file=sys.stdout, indent=0):
        self.print_code(file=file, indent=indent, language="c")

    def print_h(self, file=sys.stdout, linkage="", indent=0):
        if linkage:
            linkage += " "
        if isinstance(indent, int):
            indent *= " "
        printn = partial(print, file=file, sep="")
        println = partial(printn, indent)

        for name, function in self.functions.items():
            link = (linkage if function.linkage is None else function.linkage) + " "
            args = ", ".join(" ".join(p) for p in function.args)
            println("%s%s %s (%s);" % (linkage, function.retType, name, args))


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
        return "%s%s" % (
            self.__class__.__name__,
            (self.nLookups, self.nExtraOps, self.cost),
        )


def typeWidth(typ):
    """
    >>> typeWidth('int8_t')
    8
    >>> typeWidth('uint32_t')
    32
    >>> typeWidth('i8')
    8
    >>> typeWidth('u32')
    32
    """
    return int("".join([c for c in typ if c.isdigit()]))


def typeAbbr(typ):
    """
    >>> typeAbbr('int8_t')
    'i8'
    >>> typeAbbr('uint32_t')
    'u32'
    """
    return typ[0] + str(typeWidth(typ))


def fastType(typ):
    """
    >>> fastType('int8_t')
    'int8_t'
    >>> fastType('uint32_t')
    'uint32_t'
    >>> fastType('i8')
    'i8'
    >>> fastType('u32')
    'u32'
    """
    return typ
    # return typ.replace("int", "int_fast")


class InnerSolution(Solution):
    def __init__(self, layer, next, nLookups, nExtraOps, cost, bits=0):
        Solution.__init__(self, layer, next, nLookups, nExtraOps, cost)
        self.bits = bits

    def genCode(self, code, name=None, var="u", language="c"):
        inputVar = var
        if name:
            var = "u"
        expr = var

        if isinstance(language, str):
            language = languages[language]

        typ = language.type_for(self.layer.minV, self.layer.maxV)
        retType = fastType(typ)
        unitBits = self.layer.unitBits
        if not unitBits:
            expr = self.layer.data[0]
            return (retType, expr)

        shift = self.bits
        mask = (1 << shift) - 1

        if self.next:
            (_, expr) = self.next.genCode(
                code, None, "((%s)>>%d)" % (var, shift), language=language
            )

        # Generate data.

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
            assert layer.minV == 0
            for d in range(layer.maxV + 1):
                _expand(d, layers, len(layers) - 1, data)

        data = _combine(data, self.layer.unitBits)

        arrName, start = code.addArray(typ, typeAbbr(typ), data)

        # Generate expression.

        start = language.usize_literal(start) if start else None
        if expr == "0":
            index0 = ""
        elif shift == 0:
            index0 = str(expr)
        else:
            index0 = "((%s)<<%d)" % (language.as_usize(expr), shift)
        index1 = "((%s)&%s)" % (var, mask) if mask else ""
        index = language.as_usize(index0) + ("+" if index0 and index1 else "") + language.as_usize(index1)
        if unitBits >= 8:
            if start:
                index = "%s+%s" % (start, language.as_usize(index))
            expr = language.array_index(arrName, index)
        else:
            shift1 = int(round(log2(8 // unitBits)))
            mask1 = (8 // unitBits) - 1
            shift2 = int(round(log2(unitBits)))
            mask2 = (1 << unitBits) - 1
            funcBody = "(%s>>((i&%s)<<%d))&%s" % (language.array_index("a", "i>>%s" % shift1), mask1, shift2, mask2)
            funcName = code.addFunction(
                language.u8,
                "b%s" % unitBits,
                ((language.u8 + "*", "a"), (language.usize, "i")),
                funcBody,
            )
            if start:
                sliced_array = language.slice(arrName, start)
            else:
                sliced_array = language.borrow(arrName)
            expr = "%s(%s,%s)" % (funcName, sliced_array, index)

        # Wrap up.

        if name:
            funcName = code.addFunction(retType, name, ((language.u32, "u"),), expr)
            expr = "%s(%s)" % (funcName, inputVar)

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
    if bits <= 1:
        data = _combine2(data, lambda a, b: (b << 1) | a)
    if bits <= 2:
        data = _combine2(data, lambda a, b: (b << 2) | a)
    if bits <= 4:
        data = _combine2(data, lambda a, b: (b << 4) | a)
    return data


def _combine2(data, f):
    data2 = []
    it = iter(data)
    for first in it:
        data2.append(f(first, next(it, 0)))
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
        self.minV = min(data)
        # TODO When to check minV is zero?  Enforce if unitBits < 8
        self.unitBits = binaryBitsFor(self.minV, self.maxV)
        self.extraOps = subByteAccessOps if self.unitBits < 8 else 0
        self.bytes = ceil(self.unitBits * len(self.data) / 8)

        if self.maxV == 0:
            return

        self.split()

        solution = InnerSolution(
            self, None, 0 if self.maxV == 0 else 1, self.extraOps, self.bytes
        )
        self.solutions.append(solution)

        bits = 1
        layer = self.next
        while layer is not None:
            extraCost = ceil((layer.maxV + 1) * (1 << bits) * self.unitBits / 8)

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
        data2 = _combine2(self.data, lambda a, b: mapping[(a, b)])

        self.next = InnerLayer(data2)

    def prune_solutions(self):
        """Remove dominated solutions."""

        # Doing it the slowest, O(N^2), way for now.
        sols = self.solutions
        for a in sols:
            if a.cost == None:
                continue
            for b in sols:
                if a is b:
                    continue
                if b.cost == None:
                    continue

                # Rules of dominance: a being not worse than b
                if a.nLookups <= b.nLookups and a.fullCost <= b.fullCost:
                    b.cost = None
                    continue

        self.solutions = [s for s in self.solutions if s.cost is not None]
        self.solutions.sort(key=lambda s: s.nLookups)


class OuterSolution(Solution):
    def __init__(self, layer, next, nLookups, nExtraOps, cost):
        Solution.__init__(self, layer, next, nLookups, nExtraOps, cost)

    def genCode(self, code, name=None, var="u", language="c", private=True):
        inputVar = var
        if name:
            var = "u"
        expr = var

        if isinstance(language, str):
            language = languages[language]

        typ = language.type_for(self.layer.minV, self.layer.maxV)
        retType = fastType(typ)
        unitBits = self.layer.unitBits
        if not unitBits:
            assert False  # Audit this branch
            expr = self.layer.data[0]
            return (retType, expr)

        if self.next:
            (_, expr) = self.next.genCode(code, None, var, language=language)

        if self.layer.mult != 1:
            expr = "%d*%s" % (self.layer.mult, expr)
        if self.layer.bias != 0:
            if self.layer.bias < 0:
                expr = language.cast(retType, expr)
            expr = "%d+%s" % (self.layer.bias, expr)

        expr = language.tertiary(
            "%s<%s" % (var, len(self.layer.data)), expr, self.layer.default
        )
        # TODO Map default?

        if name:
            funcName = code.addFunction(
                retType, name, ((language.usize, "u"),), expr, private=private
            )
            expr = "%s(%s)" % (funcName, inputVar)

        return (retType, expr)


def gcd(lst):
    """
    >>> gcd([])
    1
    >>> gcd([48])
    48
    >>> gcd([-48])
    48
    >>> gcd([48, 60])
    12
    >>> gcd([48, 60, 6])
    6
    >>> gcd([48, 61, 6])
    1
    """
    it = iter(lst)
    x = abs(next(it, 1))
    for y in it:
        y = abs(y)
        while y:
            x, y = y, x % y
        if x == 1:
            break
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
        if type(self.minV) == int and type(self.maxV) == int:
            b = self.minV
            candidateBits = binaryBitsFor(0, self.maxV - b)
            if unitBits > candidateBits:
                unitBits = candidateBits
                bias = b

            m = gcd(data)
            candidateBits = binaryBitsFor(self.minV // m, self.maxV // m)
            if unitBits > candidateBits:
                unitBits = candidateBits
                bias = 0
                mult = m

            if b:
                m = gcd(d - b for d in data)
                candidateBits = binaryBitsFor(0, (self.maxV - b) // m)
                if unitBits > candidateBits:
                    unitBits = candidateBits
                    bias = b
                    mult = m
            data = [(d - bias) // mult for d in self.data]
            default = (self.default - bias) // mult

        self.unitBits = unitBits
        self.extraOps = subByteAccessOps if self.unitBits < 8 else 0
        self.bias = bias
        if bias:
            self.extraOps += 1
        self.mult = mult
        if mult:
            self.extraOps += 1

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
    not an integer.  If @mapping is not provided and data values are
    strings, the strings are written out verbatim.

    If mapping is not provided and values are strings, it is assumed that they
    all fit in an unsigned char.

    @default is value to be used for keys not specified in @data.  Defaults
    to zero.
    """

    # Set up mapping.  See docstring.
    if mapping is not None:
        # assert (all(isinstance(k, int) and not isinstance(v, int) for k,v in mapping.items()) or
        #        all(not isinstance(k, int) and isinstance(v, int) for k,v in mapping.items()))
        mapping2 = mapping.copy()
        for k, v in mapping.items():
            mapping2[v] = k
        mapping = mapping2
        del mapping2

    # Set up data as a list.
    if isinstance(data, dict):
        assert all(isinstance(k, int) for k, v in data.items())
        minK = min(data.keys())
        maxK = max(data.keys())
        assert minK >= 0
        data2 = [default] * (maxK + 1)
        for k, v in data.items():
            data2[k] = v
        data = data2
        del data2

    # Convert all to integers
    assert all(isinstance(v, int) for v in data) or all(
        not isinstance(v, int) for v in data
    )
    if not isinstance(data[0], int) and mapping is not None:
        data = [mapping[v] for v in data]
    if not isinstance(default, int) and mapping is not None:
        default = mapping[default]

    solutions = OuterLayer(data, default).solutions

    if compression is None:
        solutions.sort(key=lambda s: -s.fullCost)
        return solutions

    return pick_solution(solutions, compression)


def pick_solution(solutions, compression=1):
    return min(solutions, key=lambda s: s.nLookups + compression * log2(s.fullCost))


if __name__ == "__main__":
    import doctest

    sys.exit(doctest.testmod().failed)
