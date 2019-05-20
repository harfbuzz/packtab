# packTab

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

## TODO:

- Reduce code duplication between Inner/Outer genCode().
- Handle empty data array.
- Bake in width multiplier into array data if doing so doesn't enlarge
  data type.  Again, that would save ops.
- If an array is not larger than 64 bits, inline it in code directly
  as one integer.
- Currently we only cull array of defaults at the end.  Do it at
  beginning as well, and adjust split code to find optimum shift.
- Byte reuse!  Much bigger work item.
