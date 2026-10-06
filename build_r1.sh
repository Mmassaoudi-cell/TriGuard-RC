#!/bin/bash
cd "$(dirname "$0")"
python hybrid2.py > /dev/null 2>&1
python analysis.py > /dev/null 2>&1
python figures2.py 2>&1 | tail -1
python fill.py sec_results.tex.tpl abstract.tex.tpl conclusion.tex.tpl
python striptables.py
cd ../TriGuard-RC-Paper-R1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
bibtex main > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex 2>&1 | grep -E "^!|Undefined|undefined" | head -20
pdfinfo main.pdf | grep Pages
