"""
Download and extract the original FUNSD dataset (guillaumejaume.github.io/FUNSD/dataset.zip),
then parse every training/testing annotation into DocumentRecord form and write one combined
data/documents_all.json (199 docs -- FUNSD's own 149 train + 50 test, merged into a single dataset).

Marked for real-network use only (@pytest.mark.real equivalent — not covered by offline tests).
Run once; the resulting JSON file is what everything else in this package reads from.
"""
import argparse
import json
import urllib.request
import zipfile
from dataclasses import asdict
from pathlib import Path

from agentic_docs.config import DATA_DIR, DOCUMENTS_PATH, FUNSD_RAW_DIR
from agentic_docs.funsd.parse import parse_document

DATASET_URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"


def download_and_extract(force: bool = False) -> Path:
    """Download dataset.zip and extract it under data/funsd_raw/. Idempotent unless force=True."""
    raw_dir = FUNSD_RAW_DIR.parent
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "dataset.zip"
    if not FUNSD_RAW_DIR.exists() or force:
        print(f"Downloading {DATASET_URL} -> {zip_path}")
        urllib.request.urlretrieve(DATASET_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
        print(f"Extracted to {FUNSD_RAW_DIR}")
    else:
        print(f"Already present at {FUNSD_RAW_DIR} (use --force to re-download)")
    return FUNSD_RAW_DIR


def _parse_split(split_dir_name: str) -> list[dict]:
    ann_dir = FUNSD_RAW_DIR / split_dir_name / "annotations"
    records = []
    for path in sorted(ann_dir.glob("*.json")):
        annotation = json.loads(path.read_text(encoding="utf-8"))
        doc = parse_document(annotation, doc_id=path.stem)
        records.append(asdict(doc))
    return records


def build_document_dataset() -> None:
    """Parse the raw FUNSD training+testing annotations into one combined documents_all.json."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    records = _parse_split("training_data") + _parse_split("testing_data")
    DOCUMENTS_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} docs -> {DOCUMENTS_PATH}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download FUNSD and build the combined document dataset")
    parser.add_argument("--force", action="store_true", help="Re-download even if already present")
    args = parser.parse_args(argv)
    download_and_extract(force=args.force)
    build_document_dataset()


if __name__ == "__main__":
    main()
