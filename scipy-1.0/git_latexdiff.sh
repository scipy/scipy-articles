# use git-latexdiff to provide a highlighted PDF with changes
# requested for Nature Methods

git latexdiff --main paper.tex --output paper_diff.pdf 6ff8517237 master
