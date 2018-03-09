TEX = $(wildcard *.tex)

.PHONY: paper
paper: $(TEX)
	TEXINPUTS=.//: latexmk -pdf -use-make paper.tex

.PHONY: clean
clean:
	@rm -f *~
	latexmk -CA
