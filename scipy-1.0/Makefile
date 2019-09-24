TEX = $(wildcard *.tex)

.PHONY: paper
paper: $(TEX)
	TEXINPUTS=.//: latexmk -pdf -use-make paper.tex

.PHONY: covlett
covlett: $(TEX)
	TEXINPUTS=.//: latexmk -pdf -use-make covlett.tex

.PHONY: clean
clean:
	@rm -f *~
	latexmk -CA

.PHONY: word
word:
	pandoc -s paper.tex --bibliography=references.bib --csl static/nature.csl -o paper.docx
