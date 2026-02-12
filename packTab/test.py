import io
import os
import subprocess
import sys
import tempfile

import pytest

from packTab import (
    AutoMapping,
    Code,
    LanguageC,
    LanguageRust,
    binaryBitsFor,
    gcd,
    languages,
    languageClasses,
    pack_table,
    pick_solution,
    typeWidth,
    typeAbbr,
    fastType,
    _combine,
    _combine2,
    InnerLayer,
    OuterLayer,
    InnerSolution,
    OuterSolution,
)
from packTab.__main__ import main  # noqa: F401


# ── Utility functions ──────────────────────────────────────────────


class TestBinaryBitsFor:
    def test_zero(self):
        assert binaryBitsFor(0, 0) == 0

    def test_1bit(self):
        assert binaryBitsFor(0, 1) == 1

    def test_2bit(self):
        assert binaryBitsFor(0, 2) == 2
        assert binaryBitsFor(0, 3) == 2

    def test_4bit(self):
        assert binaryBitsFor(0, 4) == 4
        assert binaryBitsFor(0, 15) == 4

    def test_8bit_unsigned(self):
        assert binaryBitsFor(0, 16) == 8
        assert binaryBitsFor(0, 255) == 8

    def test_8bit_signed(self):
        assert binaryBitsFor(-128, 127) == 8
        assert binaryBitsFor(-1, 0) == 8

    def test_16bit(self):
        assert binaryBitsFor(0, 256) == 16
        assert binaryBitsFor(0, 65535) == 16
        assert binaryBitsFor(-32768, 32767) == 16

    def test_32bit(self):
        assert binaryBitsFor(0, 65536) == 32
        assert binaryBitsFor(0, 2**32 - 1) == 32
        assert binaryBitsFor(-(2**31), 2**31 - 1) == 32

    def test_64bit(self):
        assert binaryBitsFor(0, 2**32) == 64
        assert binaryBitsFor(0, 2**64 - 1) == 64
        assert binaryBitsFor(-(2**63), 2**63 - 1) == 64

    def test_non_int_returns_8(self):
        assert binaryBitsFor("a", "z") == 8
        assert binaryBitsFor(0.0, 1.0) == 8


class TestTypeWidth:
    def test_c_types(self):
        assert typeWidth("uint8_t") == 8
        assert typeWidth("int8_t") == 8
        assert typeWidth("uint16_t") == 16
        assert typeWidth("int32_t") == 32
        assert typeWidth("uint64_t") == 64

    def test_rust_types(self):
        assert typeWidth("u8") == 8
        assert typeWidth("i8") == 8
        assert typeWidth("u16") == 16
        assert typeWidth("i32") == 32
        assert typeWidth("u64") == 64


class TestTypeAbbr:
    def test_unsigned(self):
        assert typeAbbr("uint8_t") == "u8"
        assert typeAbbr("uint32_t") == "u32"

    def test_signed(self):
        assert typeAbbr("int8_t") == "i8"
        assert typeAbbr("int16_t") == "i16"


class TestFastType:
    def test_passthrough(self):
        assert fastType("uint8_t") == "uint8_t"
        assert fastType("u32") == "u32"


class TestGcd:
    def test_empty(self):
        assert gcd([]) == 1

    def test_single(self):
        assert gcd([48]) == 48

    def test_negative(self):
        assert gcd([-48]) == 48

    def test_pair(self):
        assert gcd([48, 60]) == 12

    def test_multiple(self):
        assert gcd([48, 60, 6]) == 6

    def test_coprime(self):
        assert gcd([48, 61, 6]) == 1

    def test_all_same(self):
        assert gcd([7, 7, 7]) == 7

    def test_one_is_one(self):
        assert gcd([1, 100, 200]) == 1


class TestAutoMapping:
    def test_bidirectional(self):
        m = AutoMapping()
        v = m[("a", "b")]
        assert v == 0
        assert m[0] == ("a", "b")
        assert m[("a", "b")] == 0

    def test_sequential_ids(self):
        m = AutoMapping()
        v0 = m[("x",)]
        v1 = m[("y",)]
        assert v0 == 0
        assert v1 == 1

    def test_duplicate_key(self):
        m = AutoMapping()
        v0 = m[("k",)]
        v1 = m[("k",)]
        assert v0 == v1


class TestCombine:
    def test_combine2(self):
        data = [1, 2, 3, 4]
        result = _combine2(data, lambda a, b: (b << 4) | a)
        assert result == [0x21, 0x43]

    def test_combine2_odd_length(self):
        data = [1, 2, 3]
        result = _combine2(data, lambda a, b: (b << 4) | a)
        assert result == [0x21, 0x03]

    def test_combine_1bit(self):
        data = [0, 1, 1, 0, 1, 0, 0, 1]
        result = _combine(data, 1)
        assert len(result) == 1

    def test_combine_2bit(self):
        data = [0, 1, 2, 3]
        result = _combine(data, 2)
        assert len(result) == 1

    def test_combine_4bit(self):
        data = [5, 10, 3, 7]
        result = _combine(data, 4)
        assert len(result) == 2

    def test_combine_8bit_noop(self):
        data = [100, 200, 50]
        result = _combine(data, 8)
        assert result == data


# ── Language backends ──────────────────────────────────────────────


class TestLanguageC:
    def setup_method(self):
        self.c = LanguageC()

    def test_type_for_unsigned(self):
        assert self.c.type_for(0, 0) == "uint8_t"
        assert self.c.type_for(0, 255) == "uint8_t"
        assert self.c.type_for(0, 256) == "uint16_t"
        assert self.c.type_for(0, 65535) == "uint16_t"
        assert self.c.type_for(0, 65536) == "uint32_t"
        assert self.c.type_for(0, 2**32 - 1) == "uint32_t"
        assert self.c.type_for(0, 2**32) == "uint64_t"

    def test_type_for_signed(self):
        assert self.c.type_for(-128, 127) == "int8_t"
        assert self.c.type_for(-32768, 32767) == "int16_t"
        assert self.c.type_for(-(2**31), 2**31 - 1) == "int32_t"
        assert self.c.type_for(-(2**63), 2**63 - 1) == "int64_t"

    def test_type_for_non_int(self):
        assert self.c.type_for("a", "z") == "uint8_t"

    def test_type_name(self):
        assert self.c.type_name("u8") == "uint8_t"
        assert self.c.type_name("i32") == "int32_t"
        assert self.c.type_name("u16") == "uint16_t"

    def test_cast(self):
        assert self.c.cast("int", "x") == "(int)(x)"

    def test_borrow(self):
        assert self.c.borrow("arr") == "arr"

    def test_slice(self):
        assert self.c.slice("arr", "5") == "arr+5"

    def test_tertiary(self):
        assert self.c.tertiary("x<10", "a", "b") == "x<10 ? a : b"

    def test_declare_array(self):
        assert (
            self.c.declare_array("static const", "uint8_t", "tbl", 4)
            == "static const uint8_t tbl[4]"
        )

    def test_declare_array_no_linkage(self):
        assert self.c.declare_array("", "uint8_t", "tbl", 4) == "uint8_t tbl[4]"

    def test_declare_function(self):
        result = self.c.declare_function(
            "static inline", "uint8_t", "lookup", (("unsigned", "u"),)
        )
        assert result == "static inline uint8_t lookup (unsigned u)"

    def test_declare_function_pointer_arg(self):
        result = self.c.declare_function(
            "", "uint8_t", "f", (("uint8_t*", "a"), ("unsigned", "i"))
        )
        assert "const uint8_t*" in result

    def test_array_index(self):
        assert self.c.array_index("arr", "i") == "arr[i]"

    def test_array_index_unsafe_ignored(self):
        c = LanguageC(unsafe_array_access=True)
        assert c.array_index("arr", "i") == "arr[i]"

    def test_as_usize(self):
        assert self.c.as_usize("expr") == "expr"

    def test_return_stmt(self):
        assert self.c.return_stmt("42") == "return 42;"

    def test_usize_literal(self):
        assert self.c.usize_literal(10) == "10u"
        assert self.c.usize_literal("") == ""

    def test_preamble(self):
        buf = io.StringIO()
        self.c.print_preamble(print=lambda *a, **kw: print(*a, file=buf, **kw))
        assert "#include <stdint.h>" in buf.getvalue()


class TestLanguageRust:
    def setup_method(self):
        self.rs = LanguageRust()

    def test_type_for_unsigned(self):
        assert self.rs.type_for(0, 0) == "u8"
        assert self.rs.type_for(0, 255) == "u8"
        assert self.rs.type_for(0, 256) == "u16"
        assert self.rs.type_for(0, 65535) == "u16"
        assert self.rs.type_for(0, 65536) == "u32"
        assert self.rs.type_for(0, 2**32 - 1) == "u32"
        assert self.rs.type_for(0, 2**32) == "u64"

    def test_type_for_signed(self):
        assert self.rs.type_for(-128, 127) == "i8"
        assert self.rs.type_for(-32768, 32767) == "i16"
        assert self.rs.type_for(-(2**31), 2**31 - 1) == "i32"
        assert self.rs.type_for(-(2**63), 2**63 - 1) == "i64"

    def test_type_name(self):
        assert self.rs.type_name("u8") == "u8"
        assert self.rs.type_name("i32") == "i32"

    def test_cast(self):
        assert self.rs.cast("u32", "x") == "(x) as u32"

    def test_borrow(self):
        assert self.rs.borrow("arr") == "&arr"

    def test_slice(self):
        assert self.rs.slice("arr", "5") == "&arr[5..]"

    def test_tertiary(self):
        assert self.rs.tertiary("x<10", "a", "b") == "if x<10 { a } else { b }"

    def test_declare_array(self):
        assert self.rs.declare_array("static", "u8", "tbl", 4) == "static tbl: [u8; 4]"

    def test_declare_function(self):
        result = self.rs.declare_function(
            "pub(crate)", "u8", "lookup", (("usize", "u"),)
        )
        assert result == "pub(crate) fn lookup (u: usize) -> u8"

    def test_declare_function_slice_arg(self):
        result = self.rs.declare_function("", "u8", "f", (("u8*", "a"), ("usize", "i")))
        assert "&[u8]" in result

    def test_array_index_safe(self):
        assert self.rs.array_index("arr", "i") == "arr[i]"

    def test_array_index_unsafe(self):
        rs = LanguageRust(unsafe_array_access=True)
        assert rs.array_index("arr", "i") == "unsafe { *(arr.get_unchecked(i)) }"

    def test_as_usize_empty(self):
        assert self.rs.as_usize("") == ""

    def test_as_usize_literal(self):
        assert self.rs.as_usize("42") == "42usize"

    def test_as_usize_variable(self):
        assert self.rs.as_usize("x+1") == "(x+1) as usize"

    def test_as_usize_parenthesized(self):
        assert self.rs.as_usize("(x+1)") == "(x+1) as usize"

    def test_return_stmt(self):
        assert self.rs.return_stmt("42") == "42"

    def test_usize_literal(self):
        assert self.rs.usize_literal(10) == "10usize"

    def test_preamble_empty(self):
        buf = io.StringIO()
        self.rs.print_preamble(print=lambda *a, **kw: print(*a, file=buf, **kw))
        assert buf.getvalue() == ""


class TestLanguagesDict:
    def test_has_c_and_rust(self):
        assert "c" in languages
        assert "rust" in languages

    def test_values_are_instances(self):
        assert isinstance(languages["c"], LanguageC)
        assert isinstance(languages["rust"], LanguageRust)

    def test_classes_dict(self):
        assert languageClasses["c"] is LanguageC
        assert languageClasses["rust"] is LanguageRust

    def test_instantiation(self):
        c = languageClasses["c"]()
        assert isinstance(c, LanguageC)
        rs = languageClasses["rust"](unsafe_array_access=True)
        assert rs.unsafe_array_access is True


# ── Code class ─────────────────────────────────────────────────────


class TestCode:
    def test_namespace(self):
        code = Code("ns")
        assert code.nameFor("foo") == "ns_foo"

    def test_add_array(self):
        code = Code("t")
        name, start = code.addArray("uint8_t", "u8", [1, 2, 3])
        assert name == "t_u8"
        assert start == 0

    def test_add_array_extends(self):
        code = Code("t")
        _, start1 = code.addArray("uint8_t", "u8", [1, 2])
        _, start2 = code.addArray("uint8_t", "u8", [3, 4])
        assert start1 == 0
        assert start2 == 2
        assert code.arrays["t_u8"].values == [1, 2, 3, 4]

    def test_add_function(self):
        code = Code("t")
        name = code.addFunction("uint8_t", "get", (("unsigned", "u"),), "u+1")
        assert name == "t_get"

    def test_add_function_dedup(self):
        code = Code("t")
        code.addFunction("uint8_t", "get", (("unsigned", "u"),), "u+1")
        name = code.addFunction("uint8_t", "get", (("unsigned", "u"),), "u+1")
        assert name == "t_get"
        assert len(code.functions) == 1

    def test_print_code_c(self):
        code = Code("t")
        code.addArray("uint8_t", "u8", [10, 20])
        code.addFunction("uint8_t", "get", (("unsigned", "u"),), "t_u8[u]")
        buf = io.StringIO()
        code.print_code(file=buf, language="c")
        output = buf.getvalue()
        assert "static const uint8_t t_u8[2]" in output
        assert "10" in output
        assert "20" in output
        assert "static inline uint8_t t_get" in output

    def test_print_code_rust(self):
        code = Code("t")
        code.addArray("u8", "u8", [10, 20])
        code.addFunction("u8", "get", (("usize", "u"),), "t_u8[u]")
        buf = io.StringIO()
        code.print_code(file=buf, language="rust")
        output = buf.getvalue()
        assert "static t_u8: [u8; 2]" in output
        assert "fn t_get" in output

    def test_print_code_public(self):
        code = Code("t")
        code.addArray("uint8_t", "u8", [1])
        buf = io.StringIO()
        code.print_code(file=buf, language="c", private=False)
        assert "extern const" in buf.getvalue()


# ── Core algorithm ─────────────────────────────────────────────────


class TestInnerLayer:
    def test_all_zeros(self):
        layer = InnerLayer([0, 0, 0, 0])
        assert layer.maxV == 0
        assert layer.minV == 0
        assert layer.unitBits == 0

    def test_simple_data(self):
        layer = InnerLayer([0, 1, 2, 3])
        assert layer.maxV == 3
        assert layer.unitBits == 2
        assert len(layer.solutions) >= 1

    def test_has_mapping_after_split(self):
        layer = InnerLayer([0, 1, 2, 3])
        assert hasattr(layer, "mapping")
        assert layer.next is not None

    def test_solutions_are_sorted(self):
        layer = InnerLayer(list(range(16)))
        lookups = [s.nLookups for s in layer.solutions]
        assert lookups == sorted(lookups)

    def test_solutions_not_dominated(self):
        layer = InnerLayer(list(range(256)))
        for a in layer.solutions:
            for b in layer.solutions:
                if a is b:
                    continue
                assert not (
                    a.nLookups <= b.nLookups and a.fullCost <= b.fullCost
                ), "Found dominated solution"


class TestOuterLayer:
    def test_strips_trailing_default(self):
        layer = OuterLayer([1, 2, 3, 0, 0, 0], 0)
        assert layer.data == [1, 2, 3]

    def test_bias_optimization(self):
        # bias gets baked in when original data fits in same type
        layer = OuterLayer([100, 101, 102, 103], 0)
        assert layer.bias == 0  # baked in: [100..103] fits in uint8_t

        # bias kept when baking in would enlarge the type
        layer = OuterLayer([1000, 1001, 1002, 1003], 0)
        assert layer.bias == 1000

    def test_gcd_optimization(self):
        # mult gets baked in when undivided data fits in same type
        layer = OuterLayer([0, 10, 20, 30], 0)
        assert layer.mult == 1  # baked in: [0,10,20,30] fits in uint8_t

        # mult kept when baking in would enlarge the type
        layer = OuterLayer([0, 128, 256, 384], 0)
        assert layer.mult == 128  # 384 needs uint16_t, 3 fits in uint8_t

    def test_has_solutions(self):
        layer = OuterLayer([1, 2, 3], 0)
        assert len(layer.solutions) >= 1
        assert all(isinstance(s, OuterSolution) for s in layer.solutions)


class TestSolution:
    def test_full_cost(self):
        layer = InnerLayer([0, 1, 2, 3])
        for s in layer.solutions:
            assert s.fullCost >= s.cost


# ── pack_table public API ─────────────────────────────────────────


class TestPackTable:
    def test_list_input(self):
        solution = pack_table([1, 2, 3, 4], default=0)
        assert isinstance(solution, (InnerSolution, OuterSolution))

    def test_dict_input(self):
        solution = pack_table({0: 5, 3: 10, 7: 15}, default=0)
        assert isinstance(solution, (InnerSolution, OuterSolution))

    def test_compression_none_returns_list(self):
        solutions = pack_table([1, 2, 3], default=0, compression=None)
        assert isinstance(solutions, list)
        assert len(solutions) >= 1

    def test_with_mapping(self):
        mapping = {"A": 0, "B": 1, "C": 2}
        solution = pack_table(["A", "B", "C", "A"], default="A", mapping=mapping)
        assert isinstance(solution, (InnerSolution, OuterSolution))

    def test_constant_data(self):
        solution = pack_table([7, 7, 7, 7], default=7)
        assert isinstance(solution, (InnerSolution, OuterSolution))

    def test_large_values(self):
        solution = pack_table([0, 1000, 2000, 3000], default=0)
        assert isinstance(solution, (InnerSolution, OuterSolution))

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            pack_table([], default=0)

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError):
            pack_table({}, default=0)


class TestPickSolution:
    def test_returns_single_solution(self):
        solutions = pack_table([1, 2, 3, 4], default=0, compression=None)
        best = pick_solution(solutions, compression=1)
        assert isinstance(best, (InnerSolution, OuterSolution))

    def test_high_compression_prefers_smaller(self):
        solutions = pack_table(list(range(64)), default=0, compression=None)
        small = pick_solution(solutions, compression=10)
        fast = pick_solution(solutions, compression=0.01)
        assert small.fullCost <= fast.fullCost or small.nLookups >= fast.nLookups


# ── End-to-end code generation ─────────────────────────────────────


def _generate(data, default=0, language="c", **lang_kwargs):
    """Helper: pack data and generate code as a string."""
    solution = pack_table(data, default=default)
    lang_cls = languageClasses[language]
    lang = lang_cls(**lang_kwargs)
    code = Code("data")
    solution.genCode(code, "get", language=lang, private=False)
    buf = io.StringIO()
    code.print_code(file=buf, language=lang)
    return buf.getvalue()


def _compile_and_run_c(c_code, data, default):
    """Compile generated C and verify every index returns the right value."""
    checks = []
    for i, v in enumerate(data):
        checks.append("  assert(data_get(%d) == %d);" % (i, v))
    for i in range(len(data), len(data) + 5):
        checks.append("  assert(data_get(%d) == %d);" % (i, default))

    full = (
        "#include <assert.h>\n"
        "#include <stdio.h>\n"
        + c_code
        + "\nint main() {\n"
        + "\n".join(checks)
        + '\n  printf("PASS\\n");\n  return 0;\n}\n'
    )

    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        f.write(full)
        src = f.name
    out = src.replace(".c", "")
    try:
        subprocess.check_call(
            ["cc", "-o", out, src, "-std=c99", "-Wall", "-Werror"],
            stderr=subprocess.PIPE,
        )
        result = subprocess.check_output([out]).decode().strip()
        assert result == "PASS"
    finally:
        os.unlink(src)
        if os.path.exists(out):
            os.unlink(out)


def _compile_and_run_rust(rs_code, data, default):
    """Compile generated Rust and verify every index returns the right value."""
    checks = []
    for i, v in enumerate(data):
        checks.append("    assert_eq!(data_get(%d) as i64, %di64);" % (i, v))
    for i in range(len(data), len(data) + 5):
        checks.append("    assert_eq!(data_get(%d) as i64, %di64);" % (i, default))

    full = (
        "#[allow(dead_code, unused_parens, overflowing_literals)]\n\n"
        + rs_code
        + "\nfn main() {\n"
        + "\n".join(checks)
        + '\n    println!("PASS");\n}\n'
    )

    with tempfile.NamedTemporaryFile(suffix=".rs", mode="w", delete=False) as f:
        f.write(full)
        src = f.name
    out = src.replace(".rs", "")
    try:
        subprocess.check_call(["rustc", "-o", out, src], stderr=subprocess.PIPE)
        result = subprocess.check_output([out]).decode().strip()
        assert result == "PASS"
    finally:
        os.unlink(src)
        if os.path.exists(out):
            os.unlink(out)


def _compile_and_run(code, data, default, language):
    """Compile and run generated code for the given language."""
    if language == "c":
        _compile_and_run_c(code, data, default)
    elif language == "rust":
        _compile_and_run_rust(code, data, default)
    else:
        raise ValueError("Unknown language: %s" % language)


@pytest.fixture(params=["c", "rust"])
def language(request):
    return request.param


class TestEndToEnd:
    """Generate code, compile, and verify correctness for both C and Rust."""

    def test_small(self, language):
        data = [1, 2, 3, 4]
        code = _generate(data, language=language)
        _compile_and_run(code, data, 0, language)

    def test_ascending(self, language):
        data = list(range(32))
        code = _generate(data, language=language)
        _compile_and_run(code, data, 0, language)

    def test_repeated_pattern(self, language):
        data = [0, 1, 2, 3] * 16
        code = _generate(data, language=language)
        _compile_and_run(code, data, 0, language)

    def test_sparse(self, language):
        data = [0] * 100
        data[7] = 42
        data[50] = 99
        data[99] = 1
        code = _generate(data, language=language)
        _compile_and_run(code, data, 0, language)

    def test_large_values(self, language):
        data = [0, 1000, 2000, 3000, 4000, 5000]
        code = _generate(data, language=language)
        _compile_and_run(code, data, 0, language)

    def test_16bit_values(self, language):
        data = [i * 100 for i in range(64)]
        code = _generate(data, language=language)
        _compile_and_run(code, data, 0, language)

    def test_nonzero_default(self, language):
        data = [5, 5, 5, 10, 5]
        code = _generate(data, default=5, language=language)
        _compile_and_run(code, data, 5, language)

    def test_256_values(self, language):
        data = list(range(256))
        code = _generate(data, language=language)
        _compile_and_run(code, data, 0, language)

    def test_constant_nonzero(self, language):
        data = [42, 42, 42, 42]
        code = _generate(data, default=0, language=language)
        _compile_and_run(code, data, 0, language)

    def test_two_values(self, language):
        data = [0, 1, 0, 1, 0, 1, 0, 1]
        code = _generate(data, language=language)
        _compile_and_run(code, data, 0, language)

    def test_gcd_bake_in(self, language):
        data = [0, 6, 12, 18, 24, 30]
        code = _generate(data, language=language)
        _compile_and_run(code, data, 0, language)

    def test_gcd_no_bake_in(self, language):
        data = [0, 128, 256, 384]
        code = _generate(data, language=language)
        _compile_and_run(code, data, 0, language)

    def test_gcd_with_bias(self, language):
        data = [100, 106, 112, 118]
        code = _generate(data, language=language)
        _compile_and_run(code, data, 0, language)

    def test_bias_bake_in(self, language):
        data = [100, 101, 102, 103, 104, 105]
        code = _generate(data, language=language)
        _compile_and_run(code, data, 0, language)

    def test_bias_no_bake_in(self, language):
        data = [1000, 1001, 1002, 1003]
        code = _generate(data, language=language)
        _compile_and_run(code, data, 0, language)

    def test_identity(self, language):
        data = list(range(64))
        code = _generate(data, language=language)
        _compile_and_run(code, data, 0, language)

    def test_identity_with_exceptions(self, language):
        data = list(range(32))
        data[10] = 99
        data[20] = 200
        code = _generate(data, language=language)
        _compile_and_run(code, data, 0, language)

    def test_identity_negative_deltas(self, language):
        data = [0, 1, 2, 3, 5, 4, 6, 7]
        code = _generate(data, language=language)
        _compile_and_run(code, data, 0, language)

    def test_identity_large_mirroring(self, language):
        data = list(range(256))
        data[40] = 41
        data[41] = 40
        data[60] = 62
        data[62] = 60
        data[91] = 93
        data[93] = 91
        code = _generate(data, language=language)
        _compile_and_run(code, data, 0, language)


class TestEndToEndRust:
    """Rust-specific code generation checks."""

    def test_safe_no_unsafe_keyword(self):
        code = _generate([1, 2, 3, 4], language="rust")
        assert "unsafe" not in code

    def test_unsafe_has_get_unchecked(self):
        code = _generate(
            [i * 7 % 256 for i in range(256)], language="rust", unsafe_array_access=True
        )
        assert "get_unchecked" in code
        assert "unsafe" in code

    def test_has_static_array(self):
        code = _generate([i * 7 % 256 for i in range(256)], language="rust")
        assert "static" in code

    def test_pub_crate_function(self):
        code = _generate([1, 2, 3], language="rust")
        assert "pub(crate)" in code

    def test_no_include(self):
        code = _generate([1, 2, 3], language="rust")
        assert "#include" not in code


class TestInlining:
    """Verify that small arrays get inlined as integer constants."""

    def test_small_data_no_array(self):
        code = _generate([1, 2, 3, 4], language="c")
        assert "data_u8[" not in code
        assert "data_get" in code

    def test_small_data_no_array_rust(self):
        code = _generate([1, 2, 3, 4], language="rust")
        assert ": [u8;" not in code
        assert "data_get" in code

    def test_large_data_has_array(self):
        code = _generate([i * 7 % 256 for i in range(256)], language="c")
        assert "uint8_t" in code


class TestMultBakeIn:
    """Verify that width multiplier is baked in when type doesn't change."""

    def test_bake_in_small_gcd(self):
        data = [0, 4, 8, 12]
        layer = OuterLayer(data, 0)
        assert layer.mult == 1

    def test_no_bake_in_type_change(self):
        data = [0, 128, 256, 384]
        layer = OuterLayer(data, 0)
        assert layer.mult == 128

    def test_bake_in_no_mult_in_code(self):
        code = _generate([0, 6, 12, 18], language="c")
        assert "6*" not in code

    def test_no_bake_in_has_mult_in_code(self):
        code = _generate([0, 128, 256, 384], language="c")
        assert "128*" in code


class TestBiasBakeIn:
    """Verify that bias is baked in when type doesn't change."""

    def test_bake_in_small_bias(self):
        layer = OuterLayer([100, 101, 102, 103], 0)
        assert layer.bias == 0

    def test_no_bake_in_type_change(self):
        layer = OuterLayer([1000, 1001, 1002, 1003], 0)
        assert layer.bias == 1000

    def test_bake_in_no_bias_in_code(self):
        code = _generate([200, 201, 202, 203], language="c")
        assert "200+" not in code

    def test_no_bake_in_has_bias_in_code(self):
        code = _generate([1000, 1003, 1001, 1002], language="c")
        assert "1000+" in code


class TestIdentity:
    """Verify identity subtraction optimization for near-linear data."""

    def test_identity_chosen_for_linear_data(self):
        data = list(range(16))
        layer = OuterLayer(data, 0)
        assert layer.identity is True

    def test_identity_not_chosen_for_nonlinear(self):
        data = [0, 5, 10, 15]
        layer = OuterLayer(data, 0)
        assert layer.identity is False

    def test_identity_with_offset(self):
        data = [100 + i for i in range(8)]
        layer = OuterLayer(data, 0)
        assert layer.identity is True

    def test_identity_no_array_for_pure_identity(self):
        code = _generate(list(range(8)), language="c")
        assert "u8[" not in code

    def test_identity_in_code(self):
        data = [0, 1, 2, 5, 4, 5, 6, 7]
        code = _generate(data, language="c")
        assert "(u)+" in code


class TestStringData:
    """Verify that string data without a mapping stores identifiers verbatim."""

    def test_string_values_verbatim_in_array(self):
        data = ["CAT_A", "CAT_B", "CAT_C", "CAT_A"]
        solution = pack_table(data, default="CAT_NONE", compression=1)
        code = Code("test")
        solution.genCode(code, "lookup", language="c", private=False)
        out = io.StringIO()
        code.print_code(file=out, language="c")
        output = out.getvalue()
        assert "CAT_A" in output
        assert "CAT_B" in output
        assert "CAT_C" in output
        assert "CAT_NONE" in output

    def test_string_default_in_ternary(self):
        data = ["X", "Y"]
        solution = pack_table(data, default="Z", compression=1)
        code = Code("test")
        solution.genCode(code, "lookup", language="c", private=False)
        out = io.StringIO()
        code.print_code(file=out, language="c")
        output = out.getvalue()
        assert "Z" in output

    def test_string_with_mapping(self):
        mapping = {"CAT_A": 0, "CAT_B": 1, "CAT_C": 2}
        data = ["CAT_A", "CAT_B", "CAT_C", "CAT_A"]
        solution = pack_table(data, default="CAT_A", compression=1, mapping=mapping)
        code = Code("test")
        solution.genCode(code, "lookup", language="c", private=False)
        out = io.StringIO()
        code.print_code(file=out, language="c")
        output = out.getvalue()
        assert "CAT_A" not in output

    def test_string_no_inline(self):
        data = ["A", "B"]
        solution = pack_table(data, default="C", compression=1)
        code = Code("test")
        solution.genCode(code, "lookup", language="c", private=False)
        out = io.StringIO()
        code.print_code(file=out, language="c")
        output = out.getvalue()
        assert "u8[" in output or "u8 " in output


# ── CLI ────────────────────────────────────────────────────────────


class TestCLI:
    def _run(self, *args):
        result = subprocess.run(
            [sys.executable, "-m", "packTab", *args],
            capture_output=True,
            text=True,
        )
        return result

    def test_no_args_shows_usage(self):
        r = self._run()
        assert r.returncode != 0
        assert "usage" in r.stderr.lower()

    def test_c_output(self):
        r = self._run("1", "2", "3", "4")
        assert r.returncode == 0
        assert "#include" in r.stdout
        assert "data_get" in r.stdout

    def test_rust_output(self):
        r = self._run("--rust", "1", "2", "3", "4")
        assert r.returncode == 0
        assert "fn data_get" in r.stdout
        assert "#include" not in r.stdout

    def test_language_flag(self):
        r = self._run("--language", "rust", "1", "2", "3", "4")
        assert r.returncode == 0
        assert "fn data_get" in r.stdout

    def test_rust_unsafe_output(self):
        # Use non-linear data to avoid identity opt and inlining
        args = [str(i * 7 % 256) for i in range(256)]
        r = self._run("--rust", "--unsafe", *args)
        assert r.returncode == 0
        assert "get_unchecked" in r.stdout

    def test_default_flag(self):
        r = self._run("--default", "99", "1", "2", "3")
        assert r.returncode == 0
        assert "99" in r.stdout

    def test_name_flag(self):
        r = self._run("--name", "my_table", "1", "2", "3")
        assert r.returncode == 0
        assert "my_table_get" in r.stdout

    def test_help(self):
        r = self._run("--help")
        assert r.returncode == 0
        assert "packTab" in r.stdout
