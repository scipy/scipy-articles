TEX = $(wildcard *.tex)

paper: $(TEX)
	latexmk -pdf -use-make paper.tex

clean:
	@rm *~
	latexmk -CA
