DOC=performance-review

all: ${DOC}.pdf

${DOC}.dvi: ${DOC}.tex
	latex $^
	latex $^

${DOC}.pdf: ${DOC}.dvi
	dvipdf $^

clean:
	rm -f ${DOC}.dvi ${DOC}.pdf

.PHONY: clean all

