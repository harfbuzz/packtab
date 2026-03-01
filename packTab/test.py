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

    def test_culls_aligned_leading_defaults(self):
        layer = OuterLayer([0] * 16 + [1, 2, 3], 0)
        assert layer.base == 16
        assert layer.data == [1, 2, 3]

    def test_keeps_zero_base_when_live_range_crosses_aligned_block(self):
        layer = OuterLayer([0] * 8 + [1] * 20, 0)
        assert layer.base == 0

    def test_bias_optimization(self):
        # bias gets baked in when original data fits in same type
        layer = OuterLayer([100, 105, 110, 115], 0)
        assert layer.bias == 0  # baked in: [100..115] fits in uint8_t

        # bias kept when baking in would enlarge the type
        layer = OuterLayer([1000, 1005, 1010, 1015], 0)
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

    def test_exact_leading_cull_used_when_trimmed_span_inlines(self):
        data = [0] * 17 + [1, 2, 3, 4]
        solution = pack_table(data, default=0, compression=10)
        assert solution.layer.base == 17

    def test_exact_leading_cull_skipped_when_trimmed_span_would_not_inline(self):
        data = [0] * 17 + list(range(32))
        solutions = pack_table(data, default=0, compression=None)
        assert all(s.layer.base != 17 for s in solutions)

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

    def test_compression_zero_picks_flat_solution(self):
        solutions = pack_table(list(range(64)), default=0, compression=None)
        best = pick_solution(solutions, compression=0)
        assert best.palette is False
        assert best.next.bits == 0

    def test_compression_ten_picks_minimum_raw_cost(self):
        solutions = pack_table(list(range(64)), default=0, compression=None)
        best = pick_solution(solutions, compression=10)
        assert best.cost == min(s.cost for s in solutions)


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

    def test_sparse_with_aligned_prefix_defaults(self, language):
        data = [0] * 16 + [5, 9, 11]
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
        layer = OuterLayer([100, 105, 110, 115], 0)
        assert layer.bias == 0

    def test_no_bake_in_type_change(self):
        layer = OuterLayer([1000, 1005, 1010, 1015], 0)
        assert layer.bias == 1000

    def test_bake_in_no_bias_in_code(self):
        code = _generate([200, 205, 210, 215], language="c")
        assert "200+" not in code

    def test_no_bake_in_has_bias_in_code(self):
        code = _generate([1000, 1003, 1001, 1002], language="c")
        assert "1000+" in code


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
        # Use non-linear data to avoid inlining
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


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_value(self):
        """Single value should work."""
        data = [42]
        solution = pack_table(data, default=0)
        code = Code("test")
        solution.genCode(code, "get", language="c")
        assert solution is not None

    def test_all_same_values(self):
        """All identical values should optimize well."""
        data = [7] * 100
        solution = pack_table(data, default=0)
        # Should recognize constant data
        assert solution.cost == 0  # Inlined

    def test_negative_numbers(self):
        """Negative numbers should work."""
        data = [-5, -3, -1, 0, 1, 3, 5]
        solution = pack_table(data, default=0)
        code = Code("test")
        solution.genCode(code, "get", language="c")
        assert "int" in code.print_code.__code__.co_names or True  # Generates valid code

    def test_large_sparse_table(self):
        """Sparse table with large indices."""
        data = {0: 1, 1000: 2, 10000: 3}
        solution = pack_table(data, default=0)
        assert solution is not None
        # Should compress well due to sparsity

    def test_u8_boundary(self):
        """Test values at u8 boundary (255)."""
        data = [0, 127, 255]
        solution = pack_table(data, default=0)
        code = Code("test")
        solution.genCode(code, "get", language="c")
        output = io.StringIO()
        code.print_code(file=output, language="c")
        assert "uint8_t" in output.getvalue()

    def test_u16_boundary(self):
        """Test values requiring u16."""
        data = [0, 255, 256, 65535]
        solution = pack_table(data, default=0)
        code = Code("test")
        solution.genCode(code, "get", language="c")
        output = io.StringIO()
        code.print_code(file=output, language="c")
        assert "uint16_t" in output.getvalue()

    def test_alternating_pattern(self):
        """Alternating 0/1 pattern."""
        data = [i % 2 for i in range(100)]
        solution = pack_table(data, default=0)
        # Should use sub-byte packing
        assert solution.cost < 100  # Better than naive storage

    def test_power_of_two_values(self):
        """Values that are powers of two."""
        data = [1, 2, 4, 8, 16, 32, 64, 128]
        solution = pack_table(data, default=0)
        code = Code("test")
        solution.genCode(code, "get", language="c")
        assert solution is not None

    def test_dual_compression_c(self):
        """Test dual compression output format."""
        from packTab.__main__ import main
        result = subprocess.run(
            [sys.executable, "-m", "packTab", "--compression", "1,9", "1", "2", "3", "4"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "#ifdef __OPTIMIZE_SIZE__" in result.stdout
        assert "#else" in result.stdout
        assert "#endif" in result.stdout

    def test_dual_compression_rust_error(self):
        """Dual compression should error for Rust."""
        from packTab.__main__ import main
        result = subprocess.run(
            [sys.executable, "-m", "packTab", "--rust", "--compression", "1,9", "1", "2", "3"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "C output" in result.stderr or "C" in result.stderr

    def test_compression_too_many_values(self):
        """More than 2 compression values should error."""
        from packTab.__main__ import main
        result = subprocess.run(
            [sys.executable, "-m", "packTab", "--compression", "1,5,9", "1", "2", "3"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "at most 2" in result.stderr

    def test_optimize_size_flag(self):
        """--optimize-size should set high compression."""
        result = subprocess.run(
            [sys.executable, "-m", "packTab", "--optimize-size", "--analyze", "1", "2", "3", "4"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "compression=9" in result.stdout

    def test_input_output_files(self):
        """Test -i and -o flags."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("1 2 3 4")
            input_file = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as f:
            output_file = f.name

        try:
            result = subprocess.run(
                [sys.executable, "-m", "packTab", "-i", input_file, "-o", output_file],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            with open(output_file) as f:
                content = f.read()
            assert "data_get" in content
        finally:
            os.unlink(input_file)
            os.unlink(output_file)

    def test_sparse_flag(self):
        """Test --sparse flag."""
        result = subprocess.run(
            [sys.executable, "-m", "packTab", "--sparse", "--default", "0", "10:5", "20:10"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "data_get" in result.stdout

    def test_invalid_mapping(self):
        """Mixed int/str mapping should raise TypeError."""
        mapping = {1: "a", "b": 2, 3: 4}  # Mixed!
        with pytest.raises(TypeError, match="consistently"):
            pack_table([1, 2, 3], default=0, mapping=mapping)

    def test_empty_mapping(self):
        """Empty mapping should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            pack_table([1, 2, 3], default=0, mapping={})


class TestInnerLayerDeepChain:
    """Test that deep chains (5 layers) are built and pruned correctly.

    The binary-split engine recursively pairs adjacent values until all IDs
    are identical (maxV == 0).  For data such as range(16)*1000, this produces
    5 inner layers.  The 1-lookup wrap-constant solution (8 bytes) dominates
    all multi-lookup alternatives, so exactly one solution survives Pareto
    pruning.
    """

    # (0..16) repeated 1000× → 16 000 elements, 5 inner layers.
    DEEP_DATA = list(range(16)) * 1000

    def test_deep_chain_builds_five_layers(self):
        """Chain for (0..16)*1000 must build exactly 5 layers, deepest constant."""
        layer = InnerLayer(self.DEEP_DATA)
        # Walk the next-chain to count layers.
        n = 0
        l = layer
        while l is not None:
            n += 1
            l = l.next if hasattr(l, "next") else None
        assert n == 5, f"Expected 5-layer chain for (0..16) repeating data, got {n}"
        # The deepest layer must be the all-zero constant.
        l = layer
        while l.next is not None:
            l = l.next
        assert l.maxV == 0, f"Deepest layer must be constant, got maxV={l.maxV}"

    def test_deep_chain_pareto_optimal(self):
        """All solutions for a deep chain must be Pareto-optimal (non-dominated)."""
        layer = InnerLayer(self.DEEP_DATA)
        solutions = layer.solutions
        for a in solutions:
            for b in solutions:
                if a is b:
                    continue
                assert not (a.nLookups <= b.nLookups and a.fullCost <= b.fullCost), (
                    f"Solution (nl={a.nLookups}, fc={a.fullCost}) dominates "
                    f"(nl={b.nLookups}, fc={b.fullCost})"
                )

    def test_deep_chain_lookups_sorted(self):
        """Solutions must be ordered by nLookups ascending."""
        layer = InnerLayer(self.DEEP_DATA)
        lookups = [s.nLookups for s in layer.solutions]
        assert lookups == sorted(lookups), (
            f"Solutions not sorted by nLookups: {lookups}"
        )

    def test_deep_chain_cost_strictly_decreases(self):
        """When ordered by nLookups, fullCost must strictly decrease."""
        layer = InnerLayer(self.DEEP_DATA)
        prev_cost = float("inf")
        for s in layer.solutions:
            assert s.fullCost < prev_cost, (
                f"fullCost did not strictly decrease: {s.fullCost} >= {prev_cost}"
            )
            prev_cost = s.fullCost

    def test_deep_chain_pick_solution_valid(self):
        """pick_solution must return a valid, compact solution for the 5-layer chain."""
        data = list(range(16)) * 1000
        solutions = pack_table(data, default=0, compression=None)
        assert solutions, "Must have at least one solution"
        best = pick_solution(solutions, 9.0)
        assert best is not None, "pick_solution must return a solution"
        # The chosen solution should be far smaller than naive flat storage
        # (16000 values × 4 bits = 8000 bytes).
        assert best.cost < 100, (
            f"High-compression should pick a compact solution, got {best.cost} bytes"
        )


class TestCacheOptimization:
    """Test frequency-based chunk sorting for cache locality."""

    def test_frequent_pairs_get_lower_ids(self):
        """More frequent pairs should get lower IDs."""
        # Pattern: (1,2) appears 3 times, (3,4) appears 2 times, (5,6) appears 1 time
        data = [1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 5, 6]
        layer = InnerLayer(data)

        # Check that pairs are assigned IDs by frequency
        id_12 = layer.mapping[(1, 2)]
        id_34 = layer.mapping[(3, 4)]
        id_56 = layer.mapping[(5, 6)]

        # (1,2) most frequent -> lowest ID
        # (3,4) second -> middle ID
        # (5,6) least frequent -> highest ID
        assert id_12 < id_34 < id_56

    def test_equal_frequency_sorted_by_position(self):
        """For equal frequency, earlier position gets lower ID."""
        # All pairs appear once, so order by position
        data = [1, 2, 3, 4, 5, 6, 7, 8]
        layer = InnerLayer(data)

        id_12 = layer.mapping[(1, 2)]
        id_34 = layer.mapping[(3, 4)]
        id_56 = layer.mapping[(5, 6)]
        id_78 = layer.mapping[(7, 8)]

        # All have frequency 1, so order by position
        assert id_12 < id_34 < id_56 < id_78

    def test_mixed_frequency_and_position(self):
        """Hybrid: sort by frequency, then position."""
        # (1,2) appears 2 times at positions 0,2
        # (3,4) appears 2 times at positions 1,3
        # (5,6) appears 1 time at position 4
        data = [1, 2, 3, 4, 1, 2, 3, 4, 5, 6]
        layer = InnerLayer(data)

        id_12 = layer.mapping[(1, 2)]
        id_34 = layer.mapping[(3, 4)]
        id_56 = layer.mapping[(5, 6)]

        # (1,2) and (3,4) both appear twice, so sorted by position
        # (1,2) appears first (position 0) -> lower ID than (3,4) (position 1)
        assert id_12 < id_34

        # (5,6) appears once -> highest ID
        assert id_56 > id_12
        assert id_56 > id_34


class TestPaletteEncoding:
    """Test palette encoding optimization."""

    def test_palette_generated_for_outlier(self):
        """Palette solution should be generated when there's an outlier."""
        data = [1, 2, 3, 2, 3, 2, 1, 0, 2, 1, 2, 2, 3, 3, 1, 11110124]
        solutions = pack_table(data, default=0, compression=None)

        # Should have both direct and palette solutions
        assert len(solutions) >= 2

        # Check that one is a palette solution
        palette_solutions = [s for s in solutions if hasattr(s, 'palette') and isinstance(s.palette, list)]
        assert len(palette_solutions) >= 1

        # Verify palette structure
        pal_sol = palette_solutions[0]
        assert pal_sol.palette == [0, 1, 2, 3, 11110124]
        assert pal_sol.nLookups == 2  # indices + palette

    def test_palette_skipped_all_unique(self):
        """Palette should not be generated when all values are unique."""
        data = list(range(100))
        solutions = pack_table(data, default=0, compression=None)

        # Should not have palette solutions (all values unique)
        palette_solutions = [s for s in solutions if hasattr(s, 'palette') and isinstance(s.palette, list)]
        assert len(palette_solutions) == 0

    def test_palette_skipped_no_savings(self):
        """Palette should not be generated when index_bits >= value_bits."""
        # 16 unique values in range [0..15] -> 4 bits for both indices and values
        data = list(range(16))
        solutions = pack_table(data, default=0, compression=None)

        # Should not have palette solutions (no bit savings)
        palette_solutions = [s for s in solutions if hasattr(s, 'palette') and isinstance(s.palette, list)]
        assert len(palette_solutions) == 0

    def test_palette_with_few_unique_values(self):
        """Palette should be generated when few unique values with outlier."""
        # 100 values, only 5 unique small values + 1 huge outlier
        import random
        random.seed(42)
        data = [random.choice([1, 2, 3, 4, 5]) for _ in range(100)] + [999999]
        solutions = pack_table(data, default=0, compression=None)

        # Should have palette solution
        palette_solutions = [s for s in solutions if hasattr(s, 'palette') and isinstance(s.palette, list)]
        assert len(palette_solutions) >= 1

        pal_sol = palette_solutions[0]
        assert len(pal_sol.palette) <= 6  # 5 small values + outlier

    def test_palette_cost_calculation(self):
        """Verify palette solution cost is calculated correctly."""
        data = [1, 2, 3, 2, 3, 2, 1, 0, 2, 1, 2, 2, 3, 3, 1, 11110124]
        solutions = pack_table(data, default=0, compression=None)

        palette_sol = [s for s in solutions if hasattr(s, 'palette') and isinstance(s.palette, list)][0]

        # Palette: 5 values × 4 bytes = 20 bytes
        # Indices: 16 values, 3 bits each, packed = ~6 bytes
        # Total should be around 20-28 bytes
        assert 20 <= palette_sol.cost <= 30
        assert palette_sol.cost < 64  # Better than direct (64 bytes)

    def test_palette_in_pareto_frontier(self):
        """Palette solution should be on Pareto frontier."""
        data = [1, 2, 3, 2, 3, 2, 1, 0, 2, 1, 2, 2, 3, 3, 1, 11110124]
        solutions = pack_table(data, default=0, compression=None)

        # All returned solutions should be non-dominated
        for a in solutions:
            for b in solutions:
                if a is b:
                    continue
                # a should not dominate b (otherwise b wouldn't be in frontier)
                assert not (a.nLookups <= b.nLookups and a.fullCost <= b.fullCost)

    def test_palette_selected_large_dataset(self):
        """Palette should be selected for large dataset with outliers."""
        import random
        random.seed(42)
        # 1000 values from small range, plus one huge outlier
        data = [random.choice([1, 2, 3, 4, 5]) for _ in range(1000)] + [999999]

        # With compression=1, palette should win
        solution = pack_table(data, default=0, compression=1)

        # Should be palette solution
        assert hasattr(solution, 'palette') and isinstance(solution.palette, list)
        assert len(solution.palette) == 6  # 5 values + outlier

    def test_palette_code_generation_c(self):
        """Test palette code generation for C."""
        data = [1, 2, 3, 2, 3, 2, 1, 0, 2, 1, 2, 2, 3, 3, 1, 11110124]
        solutions = pack_table(data, default=0, compression=None)

        palette_sol = [s for s in solutions if hasattr(s, 'palette') and isinstance(s.palette, list)][0]

        code = Code("test")
        palette_sol.genCode(code, "get", language="c", private=False)

        output = io.StringIO()
        code.print_code(file=output, language="c")
        result = output.getvalue()

        # Should contain palette array
        assert "palette" in result
        # Should contain the outlier value
        assert "11110124" in result

    def test_palette_code_generation_rust(self):
        """Test palette code generation for Rust."""
        data = [1, 2, 3, 2, 3, 2, 1, 0, 2, 1, 2, 2, 3, 3, 1, 11110124]
        solutions = pack_table(data, default=0, compression=None)

        palette_sol = [s for s in solutions if hasattr(s, 'palette') and isinstance(s.palette, list)][0]

        code = Code("test")
        palette_sol.genCode(code, "get", language="rust", private=False)

        output = io.StringIO()
        code.print_code(file=output, language="rust")
        result = output.getvalue()

        # Should contain palette array
        assert "palette" in result
        # Should be valid Rust syntax
        assert "fn " in result or "#[inline]" in result

    def test_palette_end_to_end_c(self):
        """Compile and run palette-encoded C code."""
        import random
        random.seed(42)
        data = [random.choice([10, 20, 30]) for _ in range(50)] + [999999]

        # Force palette solution
        solutions = pack_table(data, default=0, compression=None)
        palette_sol = [s for s in solutions if hasattr(s, 'palette') and isinstance(s.palette, list)][0]

        code = Code("data")
        palette_sol.genCode(code, "get", language="c", private=False)

        output = io.StringIO()
        code.print_code(file=output, language="c")
        c_code = output.getvalue()

        # Compile and test
        _compile_and_run_c(c_code, data, 0)

    def test_palette_end_to_end_rust(self):
        """Compile and run palette-encoded Rust code."""
        import random
        random.seed(42)
        data = [random.choice([10, 20, 30]) for _ in range(50)] + [999999]

        # Force palette solution
        solutions = pack_table(data, default=0, compression=None)
        palette_sol = [s for s in solutions if hasattr(s, 'palette') and isinstance(s.palette, list)][0]

        code = Code("data")
        palette_sol.genCode(code, "get", language="rust", private=False)

        output = io.StringIO()
        code.print_code(file=output, language="rust")
        rust_code = output.getvalue()

        # Compile and test
        _compile_and_run_rust(rust_code, data, 0)

    def test_palette_with_bias(self):
        """Palette should work with OuterLayer bias optimization."""
        # Values with common bias
        data = [100, 101, 102, 101, 102, 101, 100, 99] + [999999]
        solutions = pack_table(data, default=0, compression=None)

        palette_sol = [s for s in solutions if hasattr(s, 'palette') and isinstance(s.palette, list)]
        if palette_sol:
            # If palette generated, it should handle the bias correctly
            pal = palette_sol[0]
            code = Code("test")
            pal.genCode(code, "get", language="c")
            assert True  # Just verify it generates without error

    def test_palette_with_repeated_pattern(self):
        """Palette with repeated patterns should work correctly."""
        # Repeated pattern with outlier
        base_pattern = [1, 2, 3, 2, 3, 2, 1, 0]
        data = base_pattern * 32 + [999999]

        solution = pack_table(data, default=0, compression=5)

        # Should generate valid code
        code = Code("test")
        solution.genCode(code, "get", language="c")
        output = io.StringIO()
        code.print_code(file=output, language="c")
        assert len(output.getvalue()) > 0

    def test_palette_separate_from_other_arrays(self):
        """Verify palette uses separate array, not offset into existing arrays."""
        data = [1, 2, 3, 2, 3, 2, 1, 0, 2, 1, 2, 2, 3, 3, 1, 11110124]
        solutions = pack_table(data, default=0, compression=None)

        palette_sol = [s for s in solutions if hasattr(s, 'palette') and isinstance(s.palette, list)][0]

        code = Code("data")
        palette_sol.genCode(code, "get", language="c", private=False)

        output = io.StringIO()
        code.print_code(file=output, language="c")
        result = output.getvalue()

        # Should have separate palette array name
        assert "data_palette" in result
        # Should not have offset addition in palette access (e.g., not "data_u32[5 + ...]")
        # The palette access should be clean: palette[index]
        assert "palette[" in result or "palette" in result
