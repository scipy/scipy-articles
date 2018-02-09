TEX = $(wildcard *.tex)

.PHONY: paper
paper: $(TEX)
	latexmk -pdf -use-make paper.tex

.PHONY: clean
clean:
	@rm -f *~
	latexmk -CA
