#!/usr/bin/env python3
"""
update_population.py

Update the SMS population with a new extraction folder.

Reading, DOI null handling, DOI normalisation and DOI sorting are imported from clean_data.py,
so the update is processed exactly like the original population:
  - .bib -> clean_data.convert_bib_to_csvs (bibtexparser v1, entry.get('doi')) -> *_bib_converted.csv
  - csv  -> clean_data.read_csv_flexible (pandas) + clean_data.map_and_select (DOI/Document title/Abstract)
  - null DOI = clean_data.doi_null_mask, DOI key = clean_data.normalize_doi,
    order = clean_data.sort_by_doi, then keep first per DOI key.

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

Overlap between the new search and the old population is NOT a duplicate removal: those rows
are simply not added again (old 2025 entries would otherwise flag as duplicates).

Usage:
  python update_population.py extraction-17-10-26 --old ../population_11_10_26.csv --orig-raw extraction-26-11-25
"""

import argparse
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from clean_data import (convert_bib_to_csvs, doi_null_mask, guess_column, map_and_select,
                        normalize_doi, read_csv_flexible, sort_by_doi)

DATA_RAW_DIR = Path("data") / "raw"
DATA_CLEAN_DIR = Path("data") / "clean"
COLS = ["DOI", "Document title", "Abstract"]
MIN_YEAR = 2025
DBS = ("scopus", "ieee", "acm")


def txt(v):
    """Cell value as str ('' for NaN)."""
    return "" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v)


def doi_keys(series: pd.Series) -> pd.Series:
    """clean_data DOI key per row; '' where clean_data counts the DOI as null."""
    s = series.astype(object)
    return normalize_doi(s).where(~doi_null_mask(s), "")


def parse_year(raw):
    m = re.search(r"(19|20)\d{2}", txt(raw))
    return int(m.group(0)) if m else None


def file_year(name):
    m = re.search(r"_(20\d{2})(?=[_.])", name)
    return int(m.group(1)) if m else None


def bib_entry_count(path: Path):
    return len(re.findall(r"^\s*@\w+\s*\{", path.read_text(encoding="utf-8", errors="ignore"), re.M))


def load_csvs(paths):
    """Read csvs like clean_data (read_csv_flexible + map_and_select), sorted by DOI like clean_data.
    Returns list of row dicts (DOI = clean_data key, raw_DOI = as read) and per-file row counts."""
    parts, counts = [], {}
    for p in paths:
        df = read_csv_flexible(p)
        sel = map_and_select(df)
        year_c = guess_column(df.columns, ["publication year", "year"])  # only needed for the year filter
        years = [parse_year(v) for v in df[year_c]] if year_c else [None] * len(df)
        sel["year"] = [y if y is not None else file_year(p.name) for y in years]
        sel["source"] = p.name
        counts[p.name] = len(sel)
        parts.append(sel)
    if not parts:
        return [], counts
    merged = sort_by_doi(pd.concat(parts, ignore_index=True, sort=False))
    merged["raw_DOI"] = merged["DOI"]
    merged["DOI"] = doi_keys(merged["DOI"])
    rows = merged.to_dict("records")
    for r in rows:
        r["Document title"], r["Abstract"], r["raw_DOI"] = txt(r["Document title"]), txt(r["Abstract"]), txt(r["raw_DOI"])
    return rows, counts


def load_group(raw_dir: Path, prefix: str):
    """Load every population file starting with prefix; .bib files go through convert_bib_to_csvs."""
    files = sorted(p for p in raw_dir.iterdir()
                   if p.name.startswith(prefix) and p.suffix.lower() == ".csv" and p.name != "merged.csv")
    return load_csvs(files)


def is_blank(s):
    return s is None or str(s).strip() == ""


def write_csv(path: Path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).reindex(columns=cols).to_csv(path, index=False)


def split_null(rows):
    """Rows with / without a DOI (clean_data.doi_null_mask)."""
    return [r for r in rows if r["DOI"]], [r for r in rows if not r["DOI"]]


def dedupe(rows):
    """Keep first per DOI key (rows are in clean_data sort order).
    Returns (kept, dup_counter {doi: total occurrences}, removed_rows)."""
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


def db_of(name):
    """Database from a file name: acm / ieee (also 'iee') / scopus."""
    low = name.lower()
    for db, keys in (("acm", ("acm",)), ("ieee", ("ieee", "iee")), ("scopus", ("scopus",))):
        if any(k in low for k in keys):
            return db
    return "?"


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def doi_audit(raw_dois):
    """Format audit of DOI strings as read (before normalisation)."""
    vals = [txt(d) for d in raw_dois]
    filled = [d for d in vals if d.strip()]
    return (f"n={len(vals)} null={len(vals) - len(filled)} "
            f"whitespace={sum(d != d.strip() for d in filled)} "
            f"uppercase={sum(d != d.lower() for d in filled)} "
            f"url/doi:-prefix={sum(bool(re.match(r'\s*(https?://|doi:)', d, re.I)) for d in filled)} "
            f"not-starting-10.={sum(not d.strip().lower().startswith('10.') for d in filled)}")


def parse_xplore_date(s):
    for fmt in ("%d %b %Y", "%d-%b-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(txt(s).strip(), fmt)
        except ValueError:
            pass
    return None


def a_vs_original(orig_dir: Path, raw_dir: Path, old_df, old_set, a_raw, a_ok, a_null,
                  a_year, a_dedup, a_in_old, a_update, orig_date: datetime):
    """Per-database breakdown of the A update and consistency checks against the original export."""
    L = []
    # ---- original export, per db (acm = the acm_bib_converted.csv the old population was built from) ----
    orig_files = sorted(p for p in orig_dir.iterdir() if p.suffix.lower() == ".csv" and p.name != "merged.csv")
    orig_rows, orig_counts = load_csvs(orig_files)
    orig = {}
    for r in orig_rows:
        orig.setdefault(db_of(r["source"]), []).append(r)
    orig_dbs = {}  # doi -> set of original dbs
    for r in orig_rows:
        if r["DOI"]:
            orig_dbs.setdefault(r["DOI"], set()).add(db_of(r["source"]))
    orig_titles = {norm_title(r["Document title"]) for r in orig_rows} - {""}

    def tag(dbs):
        return "+".join(sorted(dbs)) if dbs else "(not in orig export)"

    L += ["", "ORIGINAL EXPORT (comparison base)",
          f"Folder: {orig_dir}  files: {orig_counts}",
          "Rows per database: " + ", ".join(f"{db}={len(orig.get(db, []))}" for db in DBS)]
    for p in sorted(orig_dir.glob("*.bib")):
        L.append(f"  {p.name}: {bib_entry_count(p)} @entries in .bib (converted csv is the source used)")

    # old population vs complete original export
    orig_multi = Counter(r["DOI"] for r in orig_rows)
    old_multi = Counter(doi_keys(old_df["DOI"]))
    only_orig = sorted(set(orig_multi) - set(old_multi) - {""})
    only_old = sorted(set(old_multi) - set(orig_multi) - {""})
    L += [f"Old population rows: {len(old_df)}  vs original export rows: {len(orig_rows)}",
          f"Old population = complete original export (same DOI multiset, nulls/dups included): "
          f"{'YES' if orig_multi == old_multi else 'NO'}",
          f"  DOIs only in original export ({len(only_orig)}): {only_orig or '(none)'}",
          f"  DOIs only in old population ({len(only_old)}): {only_old or '(none)'}",
          "  -> A is compared against every record of the original search, incl. those excluded at screening."]

    # ---- per-db pipeline for A ----
    def per(rows):
        return Counter(db_of(r["source"]) for r in rows)
    raw_c, null_c, ok_c, year_c = per(a_raw), per(a_null), per(a_ok), per(a_year)
    within = Counter()
    for db in DBS:
        within[db] = len({r["DOI"] for r in a_year if db_of(r["source"]) == db})
    first_c, inold_c = per(a_dedup), per(a_in_old)
    add25 = per(r for r in a_update if r["year"] == 2025)
    add26 = per(r for r in a_update if r["year"] and r["year"] > 2025)
    hdr = ["db", "raw", "nullDOI", "yearDrop", "kept", "uniqInDB", "firstSeen", "inOld", "add2025", "add>2025", "added"]
    tbl = [hdr]
    for db in DBS + ("TOTAL",):
        g = (lambda c: sum(c[d] for d in DBS)) if db == "TOTAL" else (lambda c, db=db: c[db])
        tbl.append([db, g(raw_c), g(null_c), g(ok_c) - g(year_c), g(year_c), g(within),
                    g(first_c), g(inold_c), g(add25), g(add26), g(add25) + g(add26)])
    w = [max(len(str(r[i])) for r in tbl) for i in range(len(hdr))]
    L += ["", "POPULATION A: PER-DATABASE BREAKDOWN",
          "raw=rows in export files (acm: rows of *_bib_converted.csv) | nullDOI=dropped, clean_data.doi_null_mask",
          "yearDrop=year<%d | kept=with DOI and year ok | uniqInDB=unique DOIs within that db (before cross-db dedup)" % MIN_YEAR,
          "firstSeen=unique after cross-db dedup (first row in clean_data DOI sort order) | inOld=already in old population",
          ""] + ["  ".join(str(c).rjust(w[i]) for i, c in enumerate(r)) for r in tbl]
    upd_dbs = {}
    for r in a_year:
        upd_dbs.setdefault(r["DOI"], set()).add(db_of(r["source"]))
    L += ["Cross-database duplicates within A (DOI found in several dbs): " +
          str(dict(sorted(Counter(tag(s) for s in upd_dbs.values() if len(s) > 1).items())) or "(none)")]

    # ---- overlap matrix ----
    m_first = Counter((tag(orig_dbs.get(r["DOI"])), db_of(r["source"])) for r in a_in_old)
    m_all = Counter((tag(orig_dbs.get(d)), tag(upd_dbs[d])) for d in {r["DOI"] for r in a_in_old})
    L += ["", f"OVERLAP WITH ORIGINAL EXPORT: (original_db, new_db) = N   [total {len(a_in_old)}]",
          "first-seen new db (sums to the overlap total):"]
    L += [f"  ({o}, {n}) = {c}" for (o, n), c in sorted(m_first.items())]
    L += ["full membership (every db the DOI appears in, both sides):"]
    L += [f"  ({o}, {n}) = {c}" for (o, n), c in sorted(m_all.items())]
    L += ["overlap by new db and year: " + str(dict(sorted(Counter((db_of(r['source']), r['year']) for r in a_in_old).items())))]

    # ---- consistency checks ----
    L += ["", "CONSISTENCY CHECKS (A vs original)",
          "DOI rule, all sides (original export, old population, update): imported from clean_data.py",
          "  read_csv_flexible + map_and_select, null = doi_null_mask, key = normalize_doi (strip + lower),",
          "  order = sort_by_doi, keep first; .bib via convert_bib_to_csvs (entry.get('doi'))",
          "DOI format audit (values as read):"]
    for db in DBS:
        L.append(f"  original {db:7} {doi_audit(r['raw_DOI'] for r in orig.get(db, []))}")
    for db in DBS:
        L.append(f"  update   {db:7} {doi_audit(r['raw_DOI'] for r in a_raw if db_of(r['source']) == db)}")
    L.append(f"  old population  {doi_audit(old_df['DOI'])}")
    L += ["Rows with empty title (map_and_select only matches a 'document title' column): " +
          ", ".join(f"original {db}={sum(is_blank(r['Document title']) for r in orig.get(db, []))}" for db in DBS) + " | " +
          ", ".join(f"update {db}={sum(is_blank(r['Document title']) for r in a_raw if db_of(r['source']) == db)}" for db in DBS)]
    for p in sorted(raw_dir.glob("population_a_update_*.bib")):
        conv = p.with_name(p.stem + "_bib_converted.csv")
        n_conv = len(read_csv_flexible(conv)) if conv.exists() else 0
        L.append(f"  {p.name}: {bib_entry_count(p)} @entries in .bib, {n_conv} rows after convert_bib_to_csvs")
    t_added = [r for r in a_update if norm_title(r["Document title"]) in orig_titles]
    L += [f"Added rows whose normalised TITLE is in the original export (possible DOI mismatch): {len(t_added)}"]
    L += [f"  - [{db_of(r['source'])}] ({r['year']}) {r['DOI']}  {r['Document title'][:90]}" for r in t_added]
    t_null = [r for r in a_null if norm_title(r["Document title"]) in orig_titles]
    L += [f"Null-DOI A rows (dropped) whose title is in the original export: {len(t_null)} of {len(a_null)}"]
    L += [f"Year filter: original export years are set by the query; update files are split per year "
          f"(2025 / 2026) and filtered year>={MIN_YEAR} (year column, else file name). Update years per db: " +
          str(dict(sorted(Counter((db_of(r['source']), r['year']) for r in a_raw).items(), key=str)))]

    # IEEE: date added to Xplore + content types
    ieee_upd = pd.concat([read_csv_flexible(p) for p in sorted(raw_dir.glob("population_a_update_*ieee*.csv"))],
                         ignore_index=True)
    ieee_orig = pd.concat([read_csv_flexible(p) for p in orig_files if db_of(p.name) == "ieee"], ignore_index=True)
    if "Date Added To Xplore" in ieee_upd.columns:
        ieee_upd["key"] = doi_keys(ieee_upd["DOI"])
        ieee_upd["pyear"] = ieee_upd["Publication Year"].map(parse_year)
        ieee_upd["added"] = ieee_upd["Date Added To Xplore"].map(parse_xplore_date)
        y25 = ieee_upd[(ieee_upd["key"] != "") & (ieee_upd["pyear"] == 2025)]
        st = Counter()
        for k, t in zip(y25["key"], y25["added"]):
            when = "no date" if t is None or pd.isna(t) else (
                "added BEFORE orig search" if t < orig_date else "added ON/AFTER orig search")
            st[("in old" if k in old_set else "new", when)] += 1
        o_dates = [t for t in ieee_orig["Date Added To Xplore"].map(parse_xplore_date) if t]
        latest = f"{max(o_dates):%Y-%m-%d}" if o_dates else "?"
        L += ["", f"IEEE 2025 records by 'Date Added To Xplore' vs original search date {orig_date:%Y-%m-%d} "
                  f"(latest date in original export: {latest})"]
        L += [f"  {k[0]:6} | {k[1]}: {n}" for k, n in sorted(st.items())]
        L += ["  'new' + 'BEFORE' = records in Xplore at search time but not returned then (same query string):",
              "  most likely metadata enriched after the original search (index terms added later); not verifiable here."]
        dt_key = "Document Identifier"
        pre = y25[[k not in old_set and t is not None and not pd.isna(t) and t < orig_date
                   for k, t in zip(y25["key"], y25["added"])]]
        L += ["IEEE content types, original: " + str(ieee_orig[dt_key].value_counts().to_dict()),
              "IEEE content types, update:   " + str(ieee_upd[dt_key].value_counts().to_dict()),
              "IEEE content types of new 2025 records added BEFORE orig search: " + str(pre[dt_key].value_counts().to_dict())]
    return L


def update_population(folder: str, old_path: Path, orig_raw: str = None, orig_date: str = "2025-11-26"):
    raw_dir = DATA_RAW_DIR / folder
    out_dir = DATA_CLEAN_DIR / folder
    if not raw_dir.is_dir():
        raise SystemExit(f"Folder not found: {raw_dir}")

    # .bib -> *_bib_converted.csv, exactly as clean_data does (only for the update folder)
    convert_bib_to_csvs(folder)

    # ---- old population ----
    old_df = read_csv_flexible(old_path)
    old_keys = doi_keys(old_df["DOI"])
    old_rows = [{c: txt(r.get(c)) for c in COLS} for r in old_df.to_dict("records")]
    old_null = int((old_keys == "").sum())
    old_dois = Counter(k for k in old_keys if k)
    old_internal_dups = {d: n for d, n in old_dois.items() if n > 1}
    old_set = set(old_dois)

    # ---- group A ----
    a_raw, a_files = load_group(raw_dir, "population_a_update")
    a_ok, a_null = split_null(a_raw)
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
    b_ok, b_null = split_null(b_raw)
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
         f"Old population null DOIs: {old_null}; internal duplicate DOIs: {len(old_internal_dups)} (old rows kept as-is in current_population)",
         f"A rows missing title/abstract: {miss(a_update,'Document title')}/{miss(a_update,'Abstract')}",
         f"B rows missing title/abstract: {miss(b_dedup,'Document title')}/{miss(b_dedup,'Abstract')}",
         f"A raw rows per file: {a_files}", f"B raw rows per file: {b_files}",
         f"A rows added per source (first-seen): {src_counts(a_update)}",
         f"B rows added per source (first-seen): {src_counts(b_added)}",
         f"A update by year: {dict(sorted(Counter(r['year'] for r in a_update).items(), key=str))}",
         ] + (a_vs_original(DATA_RAW_DIR / orig_raw, raw_dir, old_df, old_set, a_raw, a_ok, a_null, a_year,
                            a_dedup, a_in_old, a_update, datetime.strptime(orig_date, "%Y-%m-%d"))
              if orig_raw else []) + [
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
    ap.add_argument("folder", help="Subfolder in data/raw/, e.g. extraction-17-10-26")
    ap.add_argument("--old", required=True, type=Path, help="Path to the old population csv")
    ap.add_argument("--orig-raw", help="Subfolder in data/raw/ with the original population A export, "
                                       "e.g. extraction-26-11-25 (enables per-db breakdown and checks)")
    ap.add_argument("--orig-date", default="2025-11-26", help="Date of the original search (YYYY-MM-DD)")
    args = ap.parse_args()
    update_population(args.folder, args.old, args.orig_raw, args.orig_date)


if __name__ == "__main__":
    main()
