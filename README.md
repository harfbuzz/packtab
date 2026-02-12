# packTab

Pack static integer tables into compact multi-level lookup tables
to save space.  Generates C or Rust code.

## Installation

```
pip install packtab
```

## Usage

### Command line

```bash
# Generate C lookup code
python -m packTab 1 2 3 4

# Generate Rust lookup code
python -m packTab --rust 1 2 3 4

# Generate Rust with unsafe array access
python -m packTab --rust --unsafe 1 2 3 4
```

### As a library

```python
from packTab import pack_table, Code, languages

data = [0, 1, 2, 3, 0, 1, 2, 3]
solution = pack_table(data, default=0, compression=1)

code = Code("mytable")
solution.genCode(code, "lookup", language="c", private=False)
code.print_code(language="c")
```

The `pack_table` function accepts:
- A list of integers, or a dict mapping integer keys to values
- `default`: value for missing keys (default `0`)
- `compression`: tunes the size-vs-speed tradeoff (default `1`)
- `mapping`: optional mapping between string values and integers

### Rust with unsafe access

```python
from packTab import pack_table, Code, languages

data = list(range(256)) * 4
solution = pack_table(data, default=0)

lang = languages["rust"](unsafe_array_access=True)
code = Code("mytable")
solution.genCode(code, "lookup", language=lang, private=False)
code.print_code(language=lang)
```

## How it works

The algorithm builds multi-level lookup tables using dynamic programming
to find optimal split points.  Values that fit in fewer bits get packed
into sub-byte storage (1, 2, or 4 bits per item).  An outer layer applies
arithmetic reductions (GCD factoring, bias subtraction) before splitting.

The solver produces a set of Pareto-optimal solutions trading off table
size against lookup speed, and `pick_solution` selects the best one based
on the `compression` parameter.

## Testing

```bash
pytest packTab/test.py
```

## TODO

- Reduce code duplication between Inner/Outer genCode().
- Bake in width multiplier into array data if doing so doesn't enlarge
  data type.  Again, that would save ops.
- If an array is not larger than 64 bits, inline it in code directly
  as one integer.
- Currently we only cull array of defaults at the end.  Do it at
  beginning as well, and adjust split code to find optimum shift.
- Byte reuse!  Much bigger work item.

## History

I first wrote something like this back in 2001 when I needed it in FriBidi:

  https://github.com/fribidi/fribidi/blob/master/gen.tab/packtab.c

In 2019 I wanted to use that to produce more compact Unicode data tables
for HarfBuzz, but for convenience I wanted to use it from Python.  While
I considered wrapping the C code in a module, it occurred to me that I
can rewrite it in pure Python in a much cleaner way.  That code remains
a stain on my resume in terms of readability (or lack thereof!). :D

This Python version builds on the same ideas, but is different from the
C version in two major ways:

1. Whereas the C version uses backtracking to find best split opportunities,
   I found that the same can be achieved using dynamic-programming.  So the
   Python version implements the DP approach, which is much faster.

2. The C version does not try packing multiple items into a single byte.
   The Python version does.  Ie. if items fit, they might get packed into
   1, 2, or 4 bits per item.

There's also a bunch of other optimizations, which make (eventually, when
complete) the Python version more generic and usable for a wider variety
of data tables.
