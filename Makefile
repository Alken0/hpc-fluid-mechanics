pdf:
	# installation for ubuntu
	# https://gist.github.com/rain1024/98dd5e2c6c8c28f9ea9d
	# https://tex.stackexchange.com/questions/81968/sh-epspdf-command-not-found
	cd report && rm *.log *.aux *.out *.toc *.bbl *.blg || true
	cd report && pdflatex *.tex && bibtex *.aux && pdflatex *.tex && pdflatex *.tex
	cd report && rm *.log *.aux *.out *.toc *.bbl *.blg
