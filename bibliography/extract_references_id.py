"""Extract BibTeX entry IDs and save them to a CSV file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# bibtex parsing (optional dependency)
try:
	import bibtexparser
except Exception:
	bibtexparser = None


def extract_ids(bib_file: Path) -> list[str]:
	"""Return a list of BibTeX entry IDs in the order they appear."""
	if bibtexparser is None:
		raise RuntimeError("bibtexparser not available")

	with bib_file.open("r", encoding="utf-8", errors="ignore") as handle:
		bib_db = bibtexparser.load(handle)

	ids: list[str] = []
	for entry in bib_db.entries:
		entry_id = entry.get("ID") or entry.get("id")
		if entry_id:
			ids.append(str(entry_id).strip())
	return ids


def write_csv(ids: list[str], output_path: Path) -> None:
	"""Write IDs to a CSV file with a single column named tex_id."""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.writer(handle)
		writer.writerow(["tex_id"])
		for tex_id in ids:
			writer.writerow([tex_id])


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Extract BibTeX entry IDs to a CSV file."
	)
	parser.add_argument(
		"bib_file",
		type=Path,
		help="Path to the .bib file to parse.",
	)
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		default=None,
		help="Path to the output CSV file (default: <bib_file>.csv).",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	bib_file: Path = args.bib_file
	output_path: Path = args.output or bib_file.with_suffix(".csv")

	ids = extract_ids(bib_file)

	write_csv(ids, output_path)
	print(f"Wrote {len(ids)} IDs to {output_path}")


if __name__ == "__main__":
	main()
