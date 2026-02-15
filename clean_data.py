#!/usr/bin/env python3
"""
remove-duplicates.py

Combined pipeline:
- Merge all CSVs under `data/raw/<folder>` (recursively) into `merged.csv` inside that folder.
- Keep only `DOI`, `Source title`, `Abstract` columns.
- Sort merged rows by DOI (case-insensitive; missing DOIs last).
- Count null/empty DOIs and duplicate non-null DOIs.
- Produce a cleaned CSV with non-empty DOIs and deduplicated by DOI in `data/clean/<folder>/cleaned.csv`.
- Write a detailed log to `logs/<folder>/log.txt`.

Usage:
  python remove-duplicates.py <folder-name>

Example:
  python remove-duplicates.py extraction-1-19-11

This script replaces the previous separate merge and deduplicate tools.
"""

from pathlib import Path
import argparse
import pandas as pd
import sys
from datetime import datetime

# bibtex parsing (optional dependency)
try:
    import bibtexparser
except Exception:
    bibtexparser = None


DATA_RAW_DIR = Path("data") / "raw"
DATA_CLEAN_DIR = Path("data") / "clean"
LOGS_DIR = Path("data") / "clean"
COLUMN_SELECTION = ["DOI", "Document title", "Abstract"]


def find_csv_files(raw_folder: Path):
    if not raw_folder.exists() or not raw_folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {raw_folder}")
    return sorted(raw_folder.rglob("*.csv"))


def guess_column(df_cols, keywords):
    lowered = {c.lower().strip(): c for c in df_cols}
    for key in keywords:
        for low, orig in lowered.items():
            if key in low:
                return orig
    return None


def map_and_select(df: pd.DataFrame):
    doi_keys = ["doi"]
    source_keys = ["document title"]
    abstract_keys = ["abstract", "summary"]

    doi_col = guess_column(df.columns, doi_keys)
    source_col = guess_column(df.columns, source_keys)
    abstract_col = guess_column(df.columns, abstract_keys)

    result = pd.DataFrame()
    n = len(df)
    result["DOI"] = df[doi_col] if doi_col in df.columns else pd.Series([pd.NA] * n, index=df.index)
    result["Document title"] = df[source_col] if source_col in df.columns else pd.Series([pd.NA] * n, index=df.index)
    result["Abstract"] = df[abstract_col] if abstract_col in df.columns else pd.Series([pd.NA] * n, index=df.index)

    return result


def read_csv_flexible(path: Path):
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin1", low_memory=False)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV {path}: {e}")


def convert_bib_to_csvs(folder_name: str):
    """Find .bib files under data/raw/<folder_name>, parse them and write CSVs
    with columns matching COLUMN_SELECTION so they are included in the merge step.
    Requires `bibtexparser`.
    """
    base = DATA_RAW_DIR / folder_name
    if not base.exists() or not base.is_dir():
        return

    bib_files = sorted(base.rglob("*.bib"))
    if not bib_files:
        return

    if bibtexparser is None:
        raise RuntimeError("bibtexparser not available")

    for bib in bib_files:
        try:
            with open(bib, 'r', encoding='utf-8', errors='ignore') as fh:
                bib_db = bibtexparser.load(fh)
        except Exception as e:
            print(f"Warning: failed to parse {bib}: {e}", file=sys.stderr)
            continue

        rows = []
        for entry in bib_db.entries:
            doi = entry.get('doi', '') or entry.get('DOI', '')
            source = entry.get('title', '')
            abstract = entry.get('abstract', '')
            rows.append({
                'DOI': doi,
                'Document title': source,
                'Abstract': abstract,
            })

        if not rows:
            continue

        out_name = bib.with_name(bib.stem + '_bib_converted.csv')
        try:
            pd.DataFrame(rows).to_csv(out_name, index=False)
            print(f"Converted {bib} -> {out_name}")
        except Exception as e:
            print(f"Warning: failed to write CSV for {bib}: {e}", file=sys.stderr)


def merge_folder(folder_name: str) -> pd.DataFrame:
    base = DATA_RAW_DIR / folder_name
    files = find_csv_files(base)
    if not files:
        raise FileNotFoundError(f"No CSV files found under: {base}")

    parts = []
    for f in files:
        try:
            df = read_csv_flexible(f)
        except Exception as e:
            print(f"Warning: skipping {f} (read error: {e})", file=sys.stderr)
            continue
        sel = map_and_select(df)
        parts.append(sel)

    if not parts:
        raise RuntimeError("No readable CSVs produced any rows.")

    merged = pd.concat(parts, ignore_index=True, sort=False)
    # keep only requested columns (some files may miss columns)
    merged = merged.reindex(columns=COLUMN_SELECTION)

    # Sort by DOI (case-insensitive), NA last
    if "DOI" in merged.columns:
        try:
            merged = merged.sort_values(
                by="DOI",
                key=lambda s: s.fillna("").astype(str).str.lower(),
                na_position="last",
                kind="mergesort",
            ).reset_index(drop=True)
        except TypeError:
            merged = merged.sort_values(by="DOI", na_position="last", kind="mergesort").reset_index(drop=True)

    return merged


def analyze_and_clean(merged: pd.DataFrame, folder_name: str):
    # Ensure DOI column exists
    if "DOI" not in merged.columns:
        merged["DOI"] = pd.NA

    # Normalize DOI strings for counting/deduplication
    doi_series = merged["DOI"].astype(object)
    # Count null/empty DOIs
    is_null_like = doi_series.isna() | (doi_series.astype(str).str.strip() == "")
    null_doi_count = int(is_null_like.sum())

    # Non-null DOIs (as normalized lower-case strings)
    non_null_mask = ~is_null_like
    non_null = doi_series[non_null_mask].astype(str).str.strip().str.lower()

    # Count duplicates among non-null DOIs
    dup_counts = non_null.value_counts()
    duplicate_doi_values = dup_counts[dup_counts > 1].index.tolist()
    duplicate_rows_count = int((non_null.duplicated(keep='first')).sum())

    # Build cleaned DataFrame: remove rows with null/empty DOI, drop duplicates (keep first)
    cleaned = merged[non_null_mask].copy()
    # create DOI_normalized for deduplication
    cleaned["DOI_normalized"] = cleaned["DOI"].astype(str).str.strip().str.lower()
    cleaned = cleaned[~cleaned["DOI_normalized"].duplicated(keep='first')].copy()
    cleaned = cleaned.drop(columns=["DOI_normalized"])

    # Prepare directories and file paths
    merged_out = DATA_RAW_DIR / folder_name / "merged.csv"
    clean_dir = DATA_CLEAN_DIR / folder_name
    clean_dir.mkdir(parents=True, exist_ok=True)
    cleaned_out = clean_dir / "cleaned.csv"

    # Save merged and cleaned files
    merged_out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_out, index=False)
    cleaned.to_csv(cleaned_out, index=False)

    # Write log
    log_dir = LOGS_DIR / folder_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "log.txt"

    now = datetime.utcnow().isoformat() + "Z"
    initial_count = len(merged)
    final_count = len(cleaned)

    # Rows missing Document title / Abstract (from merged set)
    title_series = merged.get('Document title', pd.Series([pd.NA] * len(merged)))
    abstract_series = merged.get('Abstract', pd.Series([pd.NA] * len(merged)))
    missing_title_mask = title_series.isna() | (title_series.astype(str).str.strip() == "")
    missing_abstract_mask = abstract_series.isna() | (abstract_series.astype(str).str.strip() == "")
    missing_title_count = int(missing_title_mask.sum())
    missing_abstract_count = int(missing_abstract_mask.sum())

    log_lines = [
        f"PROCESSING LOG - {folder_name}",
        f"Timestamp: {now}",
        "",
        "SUMMARY",
        f"Initial merged rows:            {initial_count}",
        f"Null/empty DOIs:               {null_doi_count}",
        f"Duplicate rows among non-null DOI: {duplicate_rows_count}",
        f"Final cleaned rows (deduplicated): {final_count}",
        f"Rows with missing Document title:  {missing_title_count}",
        f"Rows with missing Abstract:        {missing_abstract_count}",
        "",
        "DETAILS",
        "Duplicate DOI values (occurred >1 time):",
    ]

    if duplicate_doi_values:
        for i, doi in enumerate(duplicate_doi_values, 1):
            log_lines.append(f"{i}. {doi}")
    else:
        log_lines.append("(none)")

    log_lines.append("")
    log_lines.append(f"Merged CSV written to: {merged_out}")
    log_lines.append(f"Cleaned CSV written to: {cleaned_out}")
    log_lines.append(f"Log file: {log_file}")

    log_text = "\n".join(log_lines)
    log_file.write_text(log_text, encoding="utf-8")

    # Print concise summary
    print(log_text)

    return {
        "merged_path": str(merged_out),
        "cleaned_path": str(cleaned_out),
        "log_path": str(log_file),
        "initial_count": initial_count,
        "null_doi_count": null_doi_count,
        "duplicate_rows_count": duplicate_rows_count,
        "final_count": final_count,
    }


def clean_data(folder_name: str):
    # First, convert any .bib files under the raw folder to CSVs so they are
    # picked up by the CSV merging step.
    try:
        if bibtexparser is None:
            # If there are any .bib files, inform the user to install bibtexparser
            bib_files = list((DATA_RAW_DIR / folder_name).rglob("*.bib")) if (DATA_RAW_DIR / folder_name).exists() else []
            if bib_files:
                raise SystemExit(
                    "Found .bib files but `bibtexparser` is not installed.\n"
                    "Install it with: pip install bibtexparser"
                )
        else:
            convert_bib_to_csvs(folder_name)
    except Exception as e:
        print(f"Warning while converting .bib files: {e}", file=sys.stderr)

    try:
        merged = merge_folder(folder_name)
    except Exception as e:
        print(f"Error during merge: {e}", file=sys.stderr)
        sys.exit(2)

    result = analyze_and_clean(merged, folder_name)
    print("\nData cleaning complete.")

def main():
    parser = argparse.ArgumentParser(description="Merge CSVs under data/raw/<folder>, clean duplicates, and write logs.")
    parser.add_argument("folder", help="Subfolder inside data/raw/ containing CSV files (searched recursively)")
    args = parser.parse_args()

    clean_data(args.folder)


if __name__ == "__main__":
    main()