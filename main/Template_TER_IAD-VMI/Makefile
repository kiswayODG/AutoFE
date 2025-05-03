target=main

all: ${target}.tex resources/bib/references.bib
	pdflatex ${target}.tex
	biber ${target}.bcf
	pdflatex ${target}.tex
	pdflatex ${target}.tex

clean:
	rm -f *~ ${target}.aux ${target}.log ${target}.bbl ${target}.bcf ${target}.blg ${target}.out ${target}.pdf ${target}.toc ${target}.run.xml