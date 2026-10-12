#!/usr/bin/env python3
"""
update_population.py

Update the SMS population with a new extraction folder (stdlib csv + bibtexparser, no pandas).

Inputs (data/raw/<folder>/):
  population_a_update_*  -> update search for population A (2025 + 2026 files)
  population_b_*         -> population B search (all years)
  (.csv from IEEE / Scopus / Semantic Scholar, .bib from ACM)

Outputs (data/clean/<folder>/):
  current_population.csv   old population + A update + B (DOI-unique)
  population_a_update.csv  A rows (year >= 2025) whose DOI is NOT in the old population
  population_b.csv         B rows, deduplicated by DOI within B
  new_entries.csv          rows of current_population not in the old population (+ population, year)
  population_update_log.txt

Dedup rule (same as clean_data.py): DOI stripped + lowercased, keep first, null DOIs dropped.
Overlap between the new search and the old population is NOT a duplicate removal: those rows
are simply not added again (old 2025 entries would otherwise flag as duplicates).

Usage:
  python update_population.py extraction-12-10-26 --old ../population_11_10_26.csv
"""

import argparse
import csv
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import bibtexparser
except Exception:
    bibtexparser = None

DATA_RAW_DIR = Path("data") / "raw"
DATA_CLEAN_DIR = Path("data") / "clean"
COLS = ["DOI", "Document title", "Abstract"]
MIN_YEAR = 2025
PARSE_WARNINGS = []

csv.field_size_limit(sys.maxsize)


def read_csv_flexible(path: Path):
    for enc in ("utf-8-sig", "latin1"):
        try:
            with open(path, newline="", encoding=enc) as fh:
                return list(csv.DictReader(fh))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Failed to read CSV {path}")


def guess_column(cols, keywords):
    lowered = {c.lower().strip(): c for c in cols}
    for key in keywords:
        for low, orig in lowered.items():
            if key in low:
                return orig
    return None


def norm_doi(raw):
    """Strip, lowercase, drop doi.org URL prefix. Returns (doi, was_changed_beyond_case/strip)."""
    s = (raw or "").strip()
    if s.upper() == "NA":
        return "", False
    s2 = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", s, flags=re.I).strip()
    s2 = re.sub(r"^doi:\s*", "", s2, flags=re.I).strip()
    return s2.lower(), s2 != s


def parse_year(raw):
    m = re.search(r"(19|20)\d{2}", str(raw or ""))
    return int(m.group(0)) if m else None


def rows_from_csv(path: Path):
    data = read_csv_flexible(path)
    if not data:
        return []
    cols = list(data[0].keys())
    doi_c = guess_column(cols, ["doi"])
    title_c = guess_column(cols, ["document title", "title"])
    abs_c = guess_column(cols, ["abstract", "summary"])
    year_c = guess_column(cols, ["publication year", "year"])
    file_year = None
    m = re.search(r"_(20\d{2})_", path.name)
    if m:
        file_year = int(m.group(1))
    out = []
    for r in data:
        year = parse_year(r.get(year_c)) if year_c else None
        out.append({
            "DOI": r.get(doi_c, "") if doi_c else "",
            "Document title": (r.get(title_c) or "") if title_c else "",
            "Abstract": (r.get(abs_c) or "") if abs_c else "",
            "year": year if year is not None else file_year,
            "source": path.name,
        })
    return out


def rows_from_bib(path: Path):
    if bibtexparser is None:
        raise SystemExit("Found .bib files but `bibtexparser` is not installed (pip install bibtexparser).")
    text = path.read_text(encoding="utf-8", errors="ignore")
    out = []
    if hasattr(bibtexparser, "parse_string"):  # bibtexparser v2
        lib = bibtexparser.parse_string(text)
        for e in lib.entries:
            f = e.fields_dict
            g = lambda k: f[k].value if k in f else ""
            out.append((g("doi"), g("title"), g("abstract"), g("year")))
        # malformed entries: salvage fields with regexes instead of silently losing them
        for fb in lib.failed_blocks:
            raw = getattr(fb, "raw", "") or ""
            fld = lambda k: (re.search(r"(?<![a-z])" + k + r"\s*=\s*\{(.*?)\},?\s*\n", raw, re.S | re.I) or [None, ""])[1]
            key = re.match(r"\s*@\w+\{\s*([^,\s]+),", raw)  # ACM entry keys are the DOI
            doi = fld("doi") or (key.group(1) if key and key.group(1).startswith("10.") else "")
            out.append((doi, fld("title"), fld("abstract"), fld("year")))
            PARSE_WARNINGS.append(f"{path.name}: salvaged malformed entry doi={doi or '?'}")
    else:  # v1
        db = bibtexparser.loads(text)
        for e in db.entries:
            out.append((e.get("doi", ""), e.get("title", ""), e.get("abstract", ""), e.get("year", "")))
    rows = []
    file_year = None
    m = re.search(r"_(20\d{2})_", path.name)
    if m:
        file_year = int(m.group(1))
    for doi, title, abstract, year in out:
        y = parse_year(year)
        rows.append({
            "DOI": doi,
            "Document title": re.sub(r"\s+", " ", title.replace("{", "").replace("}", "")).strip(),
            "Abstract": abstract.strip(),
            "year": y if y is not None else file_year,
            "source": path.name,
        })
    return rows


def load_group(raw_dir: Path, prefix: str):
    """Load every csv/bib starting with prefix. Returns rows, per-source raw counts."""
    rows, counts = [], {}
    files = sorted(p for p in raw_dir.iterdir()
                   if p.name.startswith(prefix) and p.suffix.lower() in (".csv", ".bib")
                   and p.name != "merged.csv")
    for p in files:
        r = rows_from_bib(p) if p.suffix.lower() == ".bib" else rows_from_csv(p)
        counts[p.name] = len(r)
        rows.extend(r)
    return rows, counts


def is_blank(s):
    return s is None or str(s).strip() == ""


def write_csv(path: Path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def prepare(rows):
    """Normalise DOIs; split null-DOI rows. Returns (with_doi, null_rows, n_prefix_fixed)."""
    good, null, fixed = [], [], 0
    for r in rows:
        doi, changed = norm_doi(r["DOI"])
        fixed += changed
        r = dict(r, DOI=doi)
        (null if not doi else good).append(r)
    return good, null, fixed


def dedupe(rows):
    """Keep first per DOI. Returns (kept, dup_counter {doi: total occurrences}, removed_rows)."""
    seen, kept, removed = set(), [], []
    occ = Counter(r["DOI"] for r in rows)
    for r in rows:
        if r["DOI"] in seen:
            removed.append(r)
        else:
            seen.add(r["DOI"])
            kept.append(r)
    return kept, {d: n for d, n in occ.items() if n > 1}, removed


def src_counts(rows):
    return dict(Counter(r["source"] for r in rows))


def update_population(folder: str, old_path: Path):
    raw_dir = DATA_RAW_DIR / folder
    out_dir = DATA_CLEAN_DIR / folder
    if not raw_dir.is_dir():
        raise SystemExit(f"Folder not found: {raw_dir}")

    # ---- old population ----
    old_rows = read_csv_flexible(old_path)
    old_rows = [{"DOI": (r.get("DOI") or "").strip(), "Document title": r.get("Document title", ""),
                 "Abstract": r.get("Abstract", "")} for r in old_rows]
    old_null = [r for r in old_rows if not r["DOI"]]
    old_dois = Counter(r["DOI"].lower() for r in old_rows if r["DOI"])
    old_internal_dups = {d: n for d, n in old_dois.items() if n > 1}
    old_set = set(old_dois)

    # ---- group A ----
    a_raw, a_files = load_group(raw_dir, "population_a_update")
    a_ok, a_null, a_fixed = prepare(a_raw)
    a_year_drop = [r for r in a_ok if r["year"] is None or r["year"] < MIN_YEAR]
    a_year = [r for r in a_ok if not (r["year"] is None or r["year"] < MIN_YEAR)]
    a_year_null = [r for r in a_null if not (r["year"] is None or r["year"] < MIN_YEAR)]
    a_dedup, a_dups, a_removed = dedupe(a_year)
    a_in_old = [r for r in a_dedup if r["DOI"] in old_set]
    a_update = [r for r in a_dedup if r["DOI"] not in old_set]
    a_update_2025 = [r for r in a_update if r["year"] == 2025]
    a_update_new = [r for r in a_update if r["year"] and r["year"] > 2025]
    a_set = {r["DOI"] for r in a_update}

    # ---- group B ----
    b_raw, b_files = load_group(raw_dir, "population_b")
    b_ok, b_null, b_fixed = prepare(b_raw)
    b_dedup, b_dups, b_removed = dedupe(b_ok)
    b_in_old = [r for r in b_dedup if r["DOI"] in old_set]
    b_in_a = [r for r in b_dedup if r["DOI"] in a_set]
    b_added = [r for r in b_dedup if r["DOI"] not in old_set and r["DOI"] not in a_set]

    # ---- current population ----
    current = [dict(r, population="old") for r in old_rows]  # old kept as-is (incl. its own dups/nulls)
    current += [dict(r, population="A") for r in a_update]
    current += [dict(r, population="B") for r in b_added]
    new_dois = [r["DOI"] for r in current if r["population"] != "old"]
    assert len(new_dois) == len(set(new_dois)), "duplicate DOIs among newly added rows"
    new_entries = [r for r in current if r["population"] != "old"]
    assert not ({r["DOI"] for r in new_entries} & old_set)
    assert len(current) == len(old_rows) + len(new_entries)

    # ---- write csvs ----
    write_csv(out_dir / "current_population.csv", current, COLS)
    write_csv(out_dir / "population_a_update.csv", a_update, COLS)
    write_csv(out_dir / "population_b.csv", b_dedup, COLS)
    write_csv(out_dir / "new_entries.csv", new_entries, COLS + ["population", "year"])

    # ---- log ----
    def miss(rows, key):
        return sum(1 for r in rows if is_blank(r[key]))

    def doi_list(d):
        return [f"{i}. {doi} (x{n})" for i, (doi, n) in enumerate(sorted(d.items()), 1)] or ["(none)"]

    def null_list(rows):
        return [f"- [{r['source']}] ({r['year']}) {r['Document title'][:110]}" for r in rows] or ["(none)"]

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    L = [f"POPULATION UPDATE LOG - {folder}", f"Timestamp: {now}",
         f"Old population: {old_path}", "", "SUMMARY",
         f"Old population rows:                                   {len(old_rows)}",
         f"Current population rows (final):                       {len(current)}",
         f"Net new entries (new_entries.csv):                     {len(new_entries)}",
         "",
         "POPULATION A UPDATE",
         f"Raw rows (all A files):                                {len(a_raw)}",
         f"Missing/null DOIs (all years):                         {len(a_null)}  (of which year>={MIN_YEAR}: {len(a_year_null)})",
         f"Dropped by year filter (< {MIN_YEAR} or no year):             {len(a_year_drop)}",
         f"Rows kept (year>={MIN_YEAR}, with DOI):                    {len(a_year)}",
         f"Duplicate removals by DOI within A:                    {len(a_removed)}  ({len(a_dups)} DOI values)",
         f"Unique A rows (year>={MIN_YEAR}):                          {len(a_dedup)}",
         f"  already in old population (overlap, not re-added):   {len(a_in_old)}",
         f"  added to population A (not in old):                  {len(a_update)}",
         f"Missing 2025 entries added (2025, not in old):         {len(a_update_2025)}",
         f"New entries added for >=2025 total:                    {len(a_update)}  (2025: {len(a_update_2025)}, >2025: {len(a_update_new)})",
         "",
         "POPULATION B",
         f"Raw rows (all B files):                                {len(b_raw)}",
         f"Missing/null DOIs:                                     {len(b_null)}",
         f"Duplicate removals by DOI within B:                    {len(b_removed)}  ({len(b_dups)} DOI values)",
         f"Unique B rows (population_b.csv):                      {len(b_dedup)}",
         f"  already in old population:                           {len(b_in_old)}",
         f"  already in A update:                                 {len(b_in_a)}",
         f"  added to current population (new):                   {len(b_added)}",
         "",
         "NEW ENTRIES BY ORIGIN",
         f"A: {len(a_update)}   B: {len(b_added)}",
         "",
         "EXTRA CHECKS",
         f"Old population null DOIs: {len(old_null)}; internal duplicate DOIs: {len(old_internal_dups)} (old rows kept as-is in current_population)",
         f"Malformed .bib entries: {PARSE_WARNINGS or '(none)'}",
         f"DOIs normalised beyond strip/lowercase (doi.org prefix etc.): A={a_fixed}, B={b_fixed}",
         f"A rows missing title/abstract: {miss(a_update,'Document title')}/{miss(a_update,'Abstract')}",
         f"B rows missing title/abstract: {miss(b_dedup,'Document title')}/{miss(b_dedup,'Abstract')}",
         f"A raw rows per file: {a_files}", f"B raw rows per file: {b_files}",
         f"A rows added per source (first-seen): {src_counts(a_update)}",
         f"B rows added per source (first-seen): {src_counts(b_added)}",
         f"A update by year: {dict(sorted(Counter(r['year'] for r in a_update).items()))}",
         "",
         "DETAILS",
         "A: missing DOIs (all years; year shown):"] + null_list(a_null) + [
         "", f"A: rows dropped by year filter ({len(a_year_drop)}):"] + [
         f"- [{r['source']}] ({r['year']}) {r['DOI']}" for r in a_year_drop] + [
         "", "A: duplicate DOI values removed within A (occurrences):"] + doi_list(a_dups) + [
         "", "A: missing 2025 entries added (DOI):"] + [f"- {r['DOI']}" for r in a_update_2025] + [
         "", "B: missing DOIs:"] + null_list(b_null) + [
         "", "B: duplicate DOI values removed within B (occurrences):"] + doi_list(b_dups) + [
         "", f"Outputs written to: {out_dir}"]
    (out_dir / "population_update_log.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:48]))


def main():
    ap = argparse.ArgumentParser(description="Update the SMS population with a new extraction folder.")
    ap.add_argument("folder", help="Subfolder in data/raw/, e.g. extraction-12-10-26")
    ap.add_argument("--old", required=True, type=Path, help="Path to the old population csv")
    args = ap.parse_args()
    update_population(args.folder, args.old)


if __name__ == "__main__":
    main()
