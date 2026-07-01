# packTab release notes

## 1.9.0

### API change

- `default=None` is now treated as an ordinary **value** (for example, mapped to
  an integer via `mapping`), restoring its use as a real sentinel such as
  HarfBuzz's "no decomposition" entry.  To request an **inferred** default from
  the boundary values, pass the new sentinel `default=NotImplemented` (list
  input only).  The default remains `0`, so existing callers are unaffected
  unless they relied on `default=None` to trigger inference (added in 1.5.0).

### Bug fixes

- Fix the inline-constant lookup path miscompiling values that need 16 bits or
  more; this could produce wrong C/Rust output at the default compression.
- Size the generated return type to also hold `default`, so negative or
  oversized defaults are returned correctly instead of wrapping (C) or failing
  to compile (Rust).
- `compression >= 10` (and `--optimize-size`) now returns the true
  minimum-byte solution; deep low-byte splits were previously pruned before
  selection.  `compression 1..9` picks are unchanged.
- Fix the palette lookup ignoring its shared-array start offset when more than
  one palette solution is emitted into a single `Code`.
- Fix odd-length non-integer (e.g. string) tables leaking the internal padding
  element as the out-of-range value instead of the default.
- Odd-length flat tables no longer emit an unreachable trailing "dead" byte.
- Repair `Code.print_h`, which previously raised `AttributeError`.
- `--analyze`'s Score column now matches the highlighted best solution (exact
  `log2` instead of a floored approximation).

Earlier history is available in the git log and release tags.
