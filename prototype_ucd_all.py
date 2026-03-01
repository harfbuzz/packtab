#!/usr/bin/env python3
"""Prototype a compact all-properties Unicode Character Database build.

This script downloads and analyzes the non-Unihan Unicode Character Database
and runs packTab over every per-codepoint property it can account for.

The prototype uses the UCD XML as the primary source, then supplements the
small set of non-Unihan properties not carried in the XML with data from the
text UCD bundle.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable
import urllib.request
import zipfile

from packTab import Code, pack_table
from packTab.ucdxml import load_ucdxml, ucdxml_get_repertoire


UNICODE_VERSION = "17.0.0"
CODEPOINT_COUNT = 0x110000
DATA_DIR = Path("data/ucd") / UNICODE_VERSION
XML_NAME = "ucd.nounihan.flat.zip"
UCD_NAME = "UCD.zip"
XML_URL = f"https://www.unicode.org/Public/{UNICODE_VERSION}/ucdxml/{XML_NAME}"
UCD_URL = f"https://www.unicode.org/Public/{UNICODE_VERSION}/ucd/{UCD_NAME}"

DEFAULT_RUNTIME_EXCLUDES = {
    "JSN",
    "Name_Alias",
    "NFKC_CF",
    "NFKC_SCF",
    "na",
    "na1",
}

DEPRECATED_PROPERTIES = {
    "FC_NFKC",
    "Gr_Link",
    "XO_NFC",
    "XO_NFD",
    "XO_NFKC",
    "XO_NFKD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--compression",
        type=float,
        default=10,
        help="packTab compression mode to use for prototype output (default: 10)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="download the required Unicode source files if they are missing",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="write the summary report as JSON",
    )
    parser.add_argument(
        "--c-out",
        type=Path,
        help="write generated C accessors for the selected properties",
    )
    parser.add_argument(
        "--only",
        help="comma-separated property short names to analyze",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="only analyze the first N properties after filtering",
    )
    parser.add_argument(
        "--profile",
        choices=("full", "runtime"),
        default="runtime",
        help="property profile to analyze (default: runtime)",
    )
    parser.add_argument(
        "--exclude",
        help="comma-separated property short names to exclude in addition to the profile",
    )
    return parser.parse_args()


def ensure_file(path: Path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with urllib.request.urlopen(url) as src, open(path, "wb") as dst:
        dst.write(src.read())


def parse_range(field: str) -> range:
    if ".." in field:
        start, end = field.split("..")
        return range(int(start, 16), int(end, 16) + 1)
    cp = int(field, 16)
    return range(cp, cp + 1)


def parse_semicolon_records(text: str):
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        yield [field.strip() for field in line.split(";")]


def load_property_aliases(ucd_zip: Path) -> dict[str, dict[str, list[str] | str]]:
    aliases = {}
    with zipfile.ZipFile(ucd_zip) as zf:
        for fields in parse_semicolon_records(zf.read("PropertyAliases.txt").decode("utf-8")):
            aliases[fields[0]] = {
                "long": fields[1],
                "aliases": fields[2:],
            }
    return aliases


def load_xml_repertoire(xml_zip: Path):
    return ucdxml_get_repertoire(load_ucdxml(xml_zip))


def xml_property_values(repertoire, prop: str) -> list[str]:
    return [entry.get(prop, "") for entry in repertoire]


def binary_property_values(ucd_zip: Path, member: str, prop_name: str) -> list[str]:
    values = ["N"] * CODEPOINT_COUNT
    with zipfile.ZipFile(ucd_zip) as zf:
        text = zf.read(member).decode("utf-8")
    for fields in parse_semicolon_records(text):
        if fields[1] != prop_name:
            continue
        for cp in parse_range(fields[0]):
            values[cp] = "Y"
    return values


def string_property_values(ucd_zip: Path, member: str, prop_name: str) -> list[str]:
    values = [""] * CODEPOINT_COUNT
    with zipfile.ZipFile(ucd_zip) as zf:
        text = zf.read(member).decode("utf-8")
    for fields in parse_semicolon_records(text):
        if fields[1] != prop_name:
            continue
        for cp in parse_range(fields[0]):
            values[cp] = fields[2]
    return values


def name_alias_values(ucd_zip: Path) -> list[str]:
    aliases = defaultdict(list)
    with zipfile.ZipFile(ucd_zip) as zf:
        text = zf.read("NameAliases.txt").decode("utf-8")
    for fields in parse_semicolon_records(text):
        cp = int(fields[0], 16)
        aliases[cp].append(f"{fields[2]}:{fields[1]}")
    values = [""] * CODEPOINT_COUNT
    for cp, vals in aliases.items():
        values[cp] = "|".join(vals)
    return values


SUPPLEMENT_LOADERS: dict[str, tuple[str, Callable[[Path], list[str]]]] = {
    "FC_NFKC": (
        "DerivedNormalizationProps.txt",
        lambda ucd_zip: string_property_values(
            ucd_zip, "DerivedNormalizationProps.txt", "FC_NFKC"
        ),
    ),
    "Gr_Link": (
        "DerivedCoreProperties.txt",
        lambda ucd_zip: binary_property_values(
            ucd_zip, "DerivedCoreProperties.txt", "Grapheme_Link"
        ),
    ),
    "Hyphen": (
        "PropList.txt",
        lambda ucd_zip: binary_property_values(ucd_zip, "PropList.txt", "Hyphen"),
    ),
    "Name_Alias": ("NameAliases.txt", name_alias_values),
    "XO_NFC": (
        "DerivedNormalizationProps.txt",
        lambda ucd_zip: binary_property_values(
            ucd_zip, "DerivedNormalizationProps.txt", "Expands_On_NFC"
        ),
    ),
    "XO_NFD": (
        "DerivedNormalizationProps.txt",
        lambda ucd_zip: binary_property_values(
            ucd_zip, "DerivedNormalizationProps.txt", "Expands_On_NFD"
        ),
    ),
    "XO_NFKC": (
        "DerivedNormalizationProps.txt",
        lambda ucd_zip: binary_property_values(
            ucd_zip, "DerivedNormalizationProps.txt", "Expands_On_NFKC"
        ),
    ),
    "XO_NFKD": (
        "DerivedNormalizationProps.txt",
        lambda ucd_zip: binary_property_values(
            ucd_zip, "DerivedNormalizationProps.txt", "Expands_On_NFKD"
        ),
    ),
}

IGNORED_UNIHAN_PROPERTIES = {
    "cjkAccountingNumeric",
    "cjkCompatibilityVariant",
    "cjkIICore",
    "cjkIRG_GSource",
    "cjkIRG_HSource",
    "cjkIRG_JSource",
    "cjkIRG_KPSource",
    "cjkIRG_KSource",
    "cjkIRG_MSource",
    "cjkIRG_SSource",
    "cjkIRG_TSource",
    "cjkIRG_UKSource",
    "cjkIRG_USource",
    "cjkIRG_VSource",
    "cjkMandarin",
    "cjkOtherNumeric",
    "cjkPrimaryNumeric",
    "cjkRSUnicode",
    "cjkTotalStrokes",
    "cjkUnihanCore2020",
}


def runtime_excludes(props: set[str]) -> set[str]:
    excluded = set(DEFAULT_RUNTIME_EXCLUDES)
    excluded.update(prop for prop in props if prop.startswith("kEH_"))
    excluded.update(prop for prop in props if prop.startswith("kTGT_"))
    excluded.update(prop for prop in props if prop.startswith("kNSHU_"))
    return excluded


def transform_bmg(values: list[str]) -> list[int]:
    data = [0] * len(values)
    for cp, value in enumerate(values):
        if value:
            data[cp] = int(value, 16) - cp
    return data


def transform_dm_hangul(values: list[str]) -> list[str]:
    out = list(values)
    for cp in range(0xAC00, 0xD7A4):
        out[cp] = ""
    return out


PROPERTY_TRANSFORMS: dict[str, tuple[str, Callable[[list], list]]] = {
    "bmg": ("delta from codepoint", transform_bmg),
    "dm": ("Hangul syllables elided algorithmically", transform_dm_hangul),
}

SHARED_STRING_POOL_CANDIDATES = {
    "FC_NFKC",
    "EqUIdeo",
    "bpb",
    "cf",
    "dm",
    "lc",
    "nv",
    "scf",
    "slc",
    "stc",
    "suc",
    "tc",
    "uc",
}


def build_packed_data(values: list) -> tuple[list[int], dict | None, object]:
    counts = Counter(values)
    default = counts.most_common(1)[0][0]
    if all(isinstance(value, int) for value in values):
        return values, None, default

    mapping: dict[object, int] = {}
    data: list[int] = []
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
        data.append(mapping[value])
    return data, mapping, default


def sanitize_symbol(prop: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in prop)


def is_fully_inlined(solution, symbol: str) -> bool:
    code = Code("probe")
    solution.genCode(code, f"{sanitize_symbol(symbol)}_get", language="c", private=False)
    return not code.arrays


def string_storage_bytes(value: str) -> int:
    return len(value.encode("utf-8")) + 1


def analyze_shared_string_pool(generated_props: list[dict[str, object]]) -> dict[str, object]:
    candidate_props = []
    local_bytes = 0
    shared_strings: set[str] = set()
    string_occurrences: defaultdict[str, set[str]] = defaultdict(set)

    for item in generated_props:
        prop = item["property"]
        mapping = item["mapping"]
        if mapping is None or prop not in SHARED_STRING_POOL_CANDIDATES:
            continue
        values = set(mapping.keys())
        candidate_props.append(prop)
        local_bytes += sum(string_storage_bytes(value) for value in values)
        shared_strings.update(values)
        for value in values:
            string_occurrences[value].add(prop)

    shared_bytes = sum(string_storage_bytes(value) for value in shared_strings)
    repeated = [
        {
            "value": value,
            "properties": sorted(props),
            "bytes": string_storage_bytes(value),
        }
        for value, props in string_occurrences.items()
        if len(props) > 1
    ]
    repeated.sort(key=lambda item: (-len(item["properties"]), -item["bytes"], item["value"]))

    return {
        "candidate_properties": sorted(candidate_props),
        "property_count": len(candidate_props),
        "local_string_bytes": local_bytes,
        "shared_string_bytes": shared_bytes,
        "potential_savings": local_bytes - shared_bytes,
        "reused_string_count": len(repeated),
        "most_reused_strings": repeated[:20],
    }


def analyze_property(
    prop: str,
    values: list,
    compression: float,
    metadata: dict[str, str],
) -> tuple[dict[str, object], dict[str, object]]:
    transform = PROPERTY_TRANSFORMS.get(prop)
    transformed_values = transform[1](values) if transform else values
    data, mapping, default = build_packed_data(transformed_values)
    packed_default = mapping[default] if mapping is not None else default
    solution = pack_table(data, default=packed_default, compression=compression)
    non_default = len(data) - transformed_values.count(default)
    default_label = "<empty>" if default == "" else default
    result = {
        "property": prop,
        "long_name": metadata.get("long", prop),
        "source": metadata["source"],
        "values": len(mapping) if mapping is not None else len(set(transformed_values)),
        "default": default_label,
        "non_default_codepoints": non_default,
        "lookups": solution.nLookups,
        "extra_ops": solution.nExtraOps,
        "bytes": solution.cost,
        "full_cost": solution.fullCost,
        "transform": transform[0] if transform else "",
        "fully_inlined": is_fully_inlined(solution, prop),
    }
    generated = {
        "property": prop,
        "symbol": sanitize_symbol(prop),
        "solution": solution,
        "mapping": mapping,
        "default": default,
    }
    return result, generated


def write_c_output(path: Path, generated_props: list[dict[str, object]], profile: str) -> None:
    code = Code("ucd")
    header_lines = [
        f"/* Unicode {UNICODE_VERSION} non-Unihan prototype ({profile}) */",
        "/* Generated by prototype_ucd_all.py */",
        "",
    ]
    for item in generated_props:
        prop = item["property"]
        symbol = item["symbol"]
        mapping = item["mapping"]
        default = item["default"]
        if mapping is None:
            header_lines.append(f"/* {prop}: direct integer property; default {default} */")
        else:
            reverse = sorted(mapping.items(), key=lambda kv: kv[1])
            preview = ", ".join(f"{idx}={value!r}" for value, idx in reverse[:12])
            if len(reverse) > 12:
                preview += ", ..."
            default_id = mapping[default]
            header_lines.append(
                f"/* {prop}: default id {default_id}; values {preview} */"
            )
        item["solution"].genCode(code, f"{symbol}_get", language="c", private=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line)
            f.write("\n")
        f.write("\n")
        code.print_code(file=f, language="c", private=False)


def collect_property_sources(
    repertoire,
    aliases: dict[str, dict[str, list[str] | str]],
) -> tuple[set[str], dict[str, dict[str, str]]]:
    xml_props = set()
    for entry in repertoire:
        xml_props.update(entry.keys())

    metadata: dict[str, dict[str, str]] = {}
    for prop in sorted(xml_props):
        long_name = aliases.get(prop, {}).get("long", prop)
        metadata[prop] = {"source": "xml", "long": str(long_name)}

    for prop in sorted(SUPPLEMENT_LOADERS):
        long_name = aliases.get(prop, {}).get("long", prop)
        metadata[prop] = {"source": SUPPLEMENT_LOADERS[prop][0], "long": str(long_name)}

    props = set(metadata) - IGNORED_UNIHAN_PROPERTIES - DEPRECATED_PROPERTIES
    return props, metadata


def main() -> int:
    args = parse_args()

    xml_zip = args.data_dir / XML_NAME
    ucd_zip = args.data_dir / UCD_NAME
    if args.download:
        ensure_file(xml_zip, XML_URL)
        ensure_file(ucd_zip, UCD_URL)
    elif not (xml_zip.exists() and ucd_zip.exists()):
        raise SystemExit(
            f"Missing UCD input files under {args.data_dir}. Re-run with --download."
        )

    repertoire = load_xml_repertoire(xml_zip)
    aliases = load_property_aliases(ucd_zip)
    props, metadata = collect_property_sources(repertoire, aliases)
    excluded = set()
    if args.profile == "runtime":
        excluded.update(runtime_excludes(props))
    if args.exclude:
        excluded.update(item.strip() for item in args.exclude.split(",") if item.strip())
    props = sorted(props - excluded)

    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        props = [prop for prop in props if prop in wanted]
    if args.limit is not None:
        props = props[: args.limit]

    results = []
    generated_props = []
    for prop in props:
        source = metadata[prop]["source"]
        if source == "xml":
            values = xml_property_values(repertoire, prop)
        else:
            values = SUPPLEMENT_LOADERS[prop][1](ucd_zip)
        result, generated = analyze_property(prop, values, args.compression, metadata[prop])
        results.append(result)
        generated_props.append(generated)
        print(
            f"{prop:12} {results[-1]['bytes']:8} bytes  "
            f"{results[-1]['lookups']} lookups  {results[-1]['values']:6} values  "
            f"{results[-1]['source']}"
        )

    total_bytes = sum(item["bytes"] for item in results)
    total_full_cost = sum(item["full_cost"] for item in results)
    inlined = [item["property"] for item in results if item["fully_inlined"]]
    summary = {
        "unicode_version": UNICODE_VERSION,
        "compression": args.compression,
        "profile": args.profile,
        "excluded_properties": sorted(excluded),
        "property_count": len(results),
        "fully_inlined_properties": inlined,
        "total_bytes": total_bytes,
        "total_full_cost": total_full_cost,
        "properties": results,
        "shared_string_pool": analyze_shared_string_pool(generated_props),
    }

    print()
    print(f"Unicode {UNICODE_VERSION} non-Unihan prototype ({args.profile})")
    print(f"Properties analyzed: {len(results)}")
    print(f"Fully inlined properties: {len(inlined)}")
    print(f"Total packed bytes: {total_bytes}")
    print(f"Total full cost: {total_full_cost}")
    pool = summary["shared_string_pool"]
    print(
        "Shared string pool candidates: "
        f"{pool['property_count']} properties, "
        f"{pool['local_string_bytes']} local bytes -> "
        f"{pool['shared_string_bytes']} shared bytes "
        f"(save {pool['potential_savings']})"
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")
    if args.c_out:
        write_c_output(args.c_out, generated_props, args.profile)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
