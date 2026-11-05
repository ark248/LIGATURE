"""Split a compiled manuscript into the RECOMB main PDF and optional appendix.

Run from paper/ after `tectonic main.tex`. Requires pypdf. The title page and
bibliography are excluded when checking the ten-page main-text limit.
"""
import argparse
from pathlib import Path

from pypdf import PdfReader, PdfWriter


def split(source, output):
    reader = PdfReader(source)
    text = [page.extract_text() or "" for page in reader.pages]
    references = next((i for i, value in enumerate(text)
                       if value.lstrip().startswith("References\n")), None)
    appendix = next((i for i, value in enumerate(text)
                     if value.lstrip().startswith("Appendix\n")), len(reader.pages))
    if references is None or not 1 < references < appendix:
        raise ValueError("Expected one title page, main text, bibliography, then optional appendix")
    body_pages = references - 1
    if body_pages > 10:
        raise ValueError(f"Main text has {body_pages} pages; RECOMB permits at most ten")
    if any(abs(float(p.mediabox.width) - 612) > .01 or
           abs(float(p.mediabox.height) - 792) > .01 for p in reader.pages):
        raise ValueError("All pages must use US Letter dimensions")
    output.mkdir(parents=True, exist_ok=True)
    spans = [("ligature_recomb2027.pdf", 0, appendix)]
    if appendix < len(reader.pages):
        spans.append(("ligature_recomb2027_supplement.pdf", appendix, len(reader.pages)))
    for name, first, last in spans:
        writer = PdfWriter()
        writer.append(reader, pages=(first, last), import_outline=False)
        writer.add_metadata({"/Title": "LIGATURE: Generative Inference of Spatial Molecular Organization from Histology",
                             "/Author": "Adhiban Arulselvan"})
        with (output / name).open("wb") as stream:
            writer.write(stream)
        print(f"{output / name}: {last-first} pages")
    print(f"Main text: {body_pages} pages; title: 1; references: {appendix-references}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("main.pdf"))
    parser.add_argument("--output-dir", type=Path, default=Path("submission"))
    args = parser.parse_args()
    split(args.input, args.output_dir)
