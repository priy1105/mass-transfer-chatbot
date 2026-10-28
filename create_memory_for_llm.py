"""Rebuild the portable reference index after changing PDFs in data/."""
from chatbot_core import ROOT, read_pdf, save_library

def main():
    paths = sorted((ROOT / "data").glob("*.pdf"))
    if not paths:
        raise SystemExit("Add at least one text-based PDF to data/ first.")
    passages = []
    for path in paths:
        passages.extend(read_pdf(path.read_bytes(), path.name))
    target = ROOT / "data" / "knowledge.json.gz"
    save_library(passages, target)
    print(f"Saved {len(passages):,} passages from {len(paths)} PDFs to {target}")

if __name__ == "__main__":
    main()
