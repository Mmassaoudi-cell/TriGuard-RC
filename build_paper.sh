#!/bin/bash
cd "$(dirname "$0")"
python aggregate.py > /dev/null 2>&1
python figures.py 2>&1 | tail -1
python fixtables.py
cd ../TriGuard-RC-Paper
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
bibtex main > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex 2>&1 | grep -E "^!|Undefined|Citation.*undefined" | head -20
pdfinfo main.pdf | grep Pages
