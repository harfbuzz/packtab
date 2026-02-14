#!/usr/bin/env python3
"""Benchmark packtab on all Unicode properties.

This loads the Unicode Character Database and packs every property
(except name) to see real-world compression performance.
"""

import sys
import os
from collections import Counter
from packTab import pack_table, Code

# Common Unicode properties to test
PROPERTIES_TO_TEST = [
    "gc",  # General Category
    "ccc",  # Canonical Combining Class
    "bc",  # Bidi Class
    "dt",  # Decomposition Type
    "nt",  # Numeric Type
    "jt",  # Joining Type
    "jg",  # Joining Group
    "ea",  # East Asian Width
    "lb",  # Line Break
    "sc",  # Script
    "age",  # Age
    "blk",  # Block
    "hst",  # Hangul Syllable Type
    "isc",  # Indic Syllabic Category
    "InSC",  # Indic Syllabic Category (alternate)
    "InPC",  # Indic Positional Category
    "vo",  # Vertical Orientation
]


def analyze_property_distribution(name, data):
    """Analyze the distribution of a property."""
    non_default = [v for v in data if v != 0]
    unique_vals = set(data)

    print(f"\n  Data characteristics:")
    print(f"    Total codepoints: {len(data)}")
    print(f"    Non-zero values: {len(non_default)} ({100*len(non_default)/len(data):.1f}%)")
    print(f"    Unique values: {len(unique_vals)}")

    if len(unique_vals) <= 20:
        print(f"    Values: {sorted(unique_vals)}")

    # Check for patterns
    # Identity check
    identity_matches = sum(1 for i, v in enumerate(data) if v == i)
    if identity_matches > len(data) * 0.5:
        print(f"    Pattern: High identity correlation ({identity_matches}/{len(data)})")

    # Run length analysis
    runs = []
    if data:
        current_val = data[0]
        current_len = 1
        for val in data[1:]:
            if val == current_val:
                current_len += 1
            else:
                runs.append(current_len)
                current_val = val
                current_len = 1
        runs.append(current_len)

    avg_run = sum(runs) / len(runs) if runs else 0
    max_run = max(runs) if runs else 0
    print(f"    Run lengths: avg={avg_run:.1f}, max={max_run}")


def analyze_solutions(name, data, default=0):
    """Analyze all Pareto solutions for a property."""
    print(f"\n{'='*70}")
    print(f"Property: {name}")
    print(f"{'='*70}")

    analyze_property_distribution(name, data)

    # Get all Pareto-optimal solutions
    print(f"\n  Computing solutions...")
    solutions = pack_table(data, default=default, compression=None)

    print(f"\n  Pareto frontier: {len(solutions)} solutions")
    print(f"  {'Lookups':<10} {'ExtraOps':<10} {'Bytes':<10} {'FullCost':<10} {'Ratio':<10}")
    print(f"  {'-'*60}")

    naive_bytes = len(data)
    for sol in solutions:
        ratio = naive_bytes / max(sol.cost, 0.1)
        print(f"  {sol.nLookups:<10} {sol.nExtraOps:<10} {sol.cost:<10} {sol.fullCost:<10} {ratio:>6.2f}x")

    # Get best with default compression
    best = pack_table(data, default=default, compression=1)
    print(f"\n  Best solution (compression=1):")
    print(f"    Lookups: {best.nLookups}")
    print(f"    Extra ops: {best.nExtraOps}")
    print(f"    Storage: {best.cost} bytes")
    print(f"    Compression: {naive_bytes / max(best.cost, 1):.2f}x")

    return best, solutions


def try_ucdxml(ucdxml_path):
    """Try to load and analyze Unicode data from UCD XML."""
    try:
        from packTab.ucdxml import load_ucdxml, ucdxml_get_repertoire

        print(f"Loading Unicode data from {ucdxml_path}...")
        ucdxml = load_ucdxml(ucdxml_path)
        repertoire = ucdxml_get_repertoire(ucdxml)

        # Build property mappings
        results = {}

        for prop in PROPERTIES_TO_TEST:
            print(f"\n\nProcessing property: {prop}")

            # Extract property values
            values_by_cp = {}
            mapping = {}
            next_id = 0

            for cp, char_data in enumerate(repertoire):
                if char_data is None:
                    continue
                if prop not in char_data:
                    continue

                val = char_data[prop]
                if val not in mapping:
                    mapping[val] = next_id
                    next_id += 1

                values_by_cp[cp] = mapping[val]

            if not values_by_cp:
                print(f"  Skipping {prop} (no data)")
                continue

            # Create dense array
            max_cp = max(values_by_cp.keys())
            data = [0] * (max_cp + 1)
            for cp, val in values_by_cp.items():
                data[cp] = val

            best, solutions = analyze_solutions(prop, data, default=0)
            results[prop] = (best, solutions, len(mapping))

        # Summary
        print(f"\n\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"{'Property':<15} {'Values':<8} {'Lookups':<10} {'Bytes':<10} {'Ratio':<10}")
        print(f"{'-'*70}")

        for prop, (best, solutions, n_values) in sorted(results.items()):
            naive = len([cp for cp, cd in enumerate(repertoire) if cd and prop in cd])
            ratio = naive / max(best.cost, 1) if best.cost > 0 else float('inf')
            print(f"{prop:<15} {n_values:<8} {best.nLookups:<10} {best.cost:<10} {ratio:>6.2f}x")

        return results

    except ImportError:
        print("lxml not installed. Install with: pip install lxml")
        return None
    except Exception as e:
        print(f"Error loading Unicode data: {e}")
        import traceback
        traceback.print_exc()
        return None


def synthetic_benchmarks():
    """Run synthetic benchmarks if Unicode data not available."""
    print("Running synthetic benchmarks...\n")

    test_cases = [
        ("Sequential", list(range(256)), 0),
        ("Sparse (1%)", [0]*1000 + [i for i in range(10)], 0),
        ("Two values alternating", [0, 1] * 128, 0),
        ("Block structure", [0]*64 + [1]*64 + [2]*64 + [3]*64, 0),
        ("Sawtooth", [(i % 32) for i in range(512)], 0),
    ]

    results = []
    for name, data, default in test_cases:
        best, solutions = analyze_solutions(name, data, default)
        results.append((name, best, solutions))

    return results


def main():
    # Try to find UCD XML file
    ucd_paths = [
        "ucd.all.flat.zip",
        "ucd.all.grouped.zip",
        "../ucd.all.flat.zip",
        "../ucd.all.grouped.zip",
        os.path.expanduser("~/ucd.all.flat.zip"),
        os.path.expanduser("~/ucd.all.grouped.zip"),
    ]

    if len(sys.argv) > 1:
        ucd_paths.insert(0, sys.argv[1])

    ucd_found = None
    for path in ucd_paths:
        if os.path.exists(path):
            ucd_found = path
            break

    if ucd_found:
        print(f"Found UCD file: {ucd_found}\n")
        try_ucdxml(ucd_found)
    else:
        print("Unicode UCD XML file not found.")
        print("Download from: https://www.unicode.org/Public/UCD/latest/ucdxml/")
        print("Expected filename: ucd.all.flat.zip or ucd.all.grouped.zip")
        print()
        print("Running synthetic benchmarks instead...\n")
        synthetic_benchmarks()


if __name__ == "__main__":
    main()
