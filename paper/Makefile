## Makefile — compile JOSS-style paper with MacTeX
## Usage: make        (build PDF)
##        make clean  (remove auxiliary files)
##        make purge  (remove all generated files including PDF)

PAPER   := paper
LATEX   := pdflatex
BIBTEX  := bibtex
FLAGS   := -interaction=nonstopmode -halt-on-error

.PHONY: all clean purge

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex $(PAPER).bib
	$(LATEX) $(FLAGS) $(PAPER)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(FLAGS) $(PAPER)
	$(LATEX) $(FLAGS) $(PAPER)
	@echo "✓  Built $(PAPER).pdf"

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg \
	      $(PAPER).log $(PAPER).out $(PAPER).toc \
	      $(PAPER).fls $(PAPER).fdb_latexmk

purge: clean
	rm -f $(PAPER).pdf
