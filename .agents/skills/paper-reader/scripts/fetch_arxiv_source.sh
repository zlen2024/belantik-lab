#!/usr/bin/env bash
# Download an arXiv paper's PDF and LaTeX source, unpack the source, find the
# main .tex file, and convert figures to PNG so they can be embedded in HTML.
# With a paper folder, also copy the PDF and the source into that folder.
#
# Usage: fetch_arxiv_source.sh <arxiv url or id> [paper folder]
#   e.g. fetch_arxiv_source.sh https://arxiv.org/abs/2410.24164 paper-notes/pi0
#        fetch_arxiv_source.sh 2410.24164v3
#
# Cache (reused on later runs):
#   ~/.cache/arxiv-papers/<id>/source.bin    raw source download
#   ~/.cache/arxiv-papers/<id>/paper.pdf     PDF
#   ~/.cache/arxiv-papers/<id>/src/          unpacked LaTeX source
#   ~/.cache/arxiv-papers/<id>/figures/      PNG copies of every figure
# Paper folder (only when given):
#   <paper folder>/paper.pdf
#   <paper folder>/source/                   unpacked LaTeX source
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "Usage: $0 <arxiv url or id> [paper folder]" >&2
  exit 1
fi

# Accept abs/pdf/src/html URLs or a bare id; keep an explicit version suffix.
id=$(printf '%s' "$1" | sed -E \
  -e 's#^https?://(www\.|export\.)?arxiv\.org/(abs|pdf|src|html|e-print)/##' \
  -e 's#\.pdf$##' -e 's#[?\#].*$##' -e 's#/$##')
if ! printf '%s' "$id" | grep -Eq '^([0-9]{4}\.[0-9]{4,5}|[a-z-]+(\.[A-Z]{2})?/[0-9]{7})(v[0-9]+)?$'; then
  echo "Could not parse an arXiv id from: $1" >&2
  exit 1
fi
paper="${2:-}"

dir="$HOME/.cache/arxiv-papers/${id//\//_}"
mkdir -p "$dir"

if [ ! -s "$dir/paper.pdf" ]; then
  curl -fsSL -A "paper-reader-skill" -o "$dir/paper.pdf" "https://arxiv.org/pdf/$id"
  case "$(file -b "$dir/paper.pdf")" in
    PDF*) ;;
    *) rm -f "$dir/paper.pdf"; echo "Download from https://arxiv.org/pdf/$id is not a PDF" >&2; exit 1 ;;
  esac
fi

if [ ! -s "$dir/source.bin" ]; then
  curl -fsSL -A "paper-reader-skill" -o "$dir/source.bin" "https://arxiv.org/src/$id"
fi

has_source=1
if [ ! -d "$dir/src" ]; then
  mkdir -p "$dir/src"
  case "$(file -b "$dir/source.bin")" in
    gzip*)
      if tar -tzf "$dir/source.bin" >/dev/null 2>&1; then
        tar -xzf "$dir/source.bin" -C "$dir/src"
      else
        # A single gzipped .tex file, not a tarball.
        gunzip -c "$dir/source.bin" > "$dir/src/main.tex"
      fi ;;
    *tar*) tar -xf "$dir/source.bin" -C "$dir/src" ;;
    PDF*) rmdir "$dir/src"; has_source=0 ;;
    *) cp "$dir/source.bin" "$dir/src/main.tex" ;;
  esac
fi

# Copy into the paper folder without overwriting anything already there.
if [ -n "$paper" ]; then
  mkdir -p "$paper"
  [ -s "$paper/paper.pdf" ] || cp "$dir/paper.pdf" "$paper/paper.pdf"
  if [ "$has_source" = 1 ] && [ ! -d "$paper/source" ]; then
    mkdir -p "$paper/source"
    cp -R "$dir/src/." "$paper/source/"
  fi
fi

echo "ID: $id"
echo "CACHE: $dir"
[ -z "$paper" ] || echo "PAPER_FOLDER: $paper (paper.pdf$( [ "$has_source" = 1 ] && echo ', source/'))"
if [ -n "$paper" ]; then echo "PDF: $paper/paper.pdf"; else echo "PDF: $dir/paper.pdf"; fi

if [ "$has_source" = 0 ]; then
  echo "NO_SOURCE: arXiv only has a PDF for $id. Read the PDF above instead."
  exit 0
fi

# Read the source from the paper folder when there is one.
src="$dir/src"
[ -z "$paper" ] || src="$paper/source"

# The main file declares \documentclass and contains \begin{document}.
entry=""
while IFS= read -r f; do
  if grep -q '\\begin{document}' "$f"; then entry="$f"; break; fi
done < <(grep -rl --include='*.tex' '\\documentclass' "$src" 2>/dev/null || true)

# Convert figures to PNG once. For PDF figures prefer poppler, then
# Ghostscript (both honor the crop box), and fall back to macOS sips, which
# can render some figures blank or cropped.
if command -v pdftoppm >/dev/null 2>&1; then renderer=pdftoppm
elif command -v gs >/dev/null 2>&1; then renderer=gs
elif command -v sips >/dev/null 2>&1; then renderer=sips
else renderer=none
fi
pdf_to_png() {
  case "$renderer" in
    pdftoppm) pdftoppm -png -r 200 -cropbox -singlefile "$1" "${2%.png}" ;;
    gs) gs -q -dSAFER -dBATCH -dNOPAUSE -dUseCropBox -dFirstPage=1 -dLastPage=1 \
          -dTextAlphaBits=4 -dGraphicsAlphaBits=4 -r200 -sDEVICE=png16m -sOutputFile="$2" "$1" ;;
    sips) sips -s format png "$1" --out "$2" ;;
    *) return 1 ;;
  esac
}
mkdir -p "$dir/figures"
find "$dir/src" -type f \( -iname '*.pdf' -o -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) |
while IFS= read -r f; do
  rel=${f#"$dir/src/"}
  name=$(printf '%s' "${rel%.*}" | tr '/ ' '__')
  out="$dir/figures/$name.png"
  [ -s "$out" ] && continue
  case "$f" in
    *.pdf|*.PDF) pdf_to_png "$f" "$out" >/dev/null 2>&1 || true ;;
    *.png|*.PNG) cp "$f" "$out" ;;
    *) sips -s format png "$f" --out "$out" >/dev/null 2>&1 || cp "$f" "$out" ;;
  esac
done

[ -n "$entry" ] || entry='not found; look for the .tex file with \begin{document}'
echo "ENTRYPOINT: $entry"
echo "TEX_FILES:"
find "$src" -type f -name '*.tex' | sed "s#^$src/#  #" | sort
echo "FIGURES (PNG via $renderer, in $dir/figures):"
find "$dir/figures" -type f -name '*.png' | sed "s#^$dir/figures/#  #" | sort
