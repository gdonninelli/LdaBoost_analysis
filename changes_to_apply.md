### Mandatory
all text, formulas section, or elements that you add in the latex file must be in color red to do so ensure that the preamble contains:
```Latex
% --- Diff coloring preamble ---
\usepackage{xcolor}
\definecolor{RED}{rgb}{1,0,0}
\definecolor{BLUE}{rgb}{0,0,1}

\providecommand{\DIFaddtex}[1]{\textcolor{red}{#1}}
\providecommand{\DIFdeltex}[1]{} % hide deleted text

\providecommand{\DIFaddbegin}{}
\providecommand{\DIFaddend}{}
\providecommand{\DIFdelbegin}{}
\providecommand{\DIFdelend}{}

\providecommand{\DIFadd}[1]{\texorpdfstring{\DIFaddtex{#1}}{#1}}
\providecommand{\DIFdel}[1]{\texorpdfstring{\DIFdeltex{#1}}{}}
```
Then, inside the document, use these commands:
```latex
This is old text \DIFadd{this is new text in red}.
\DIFdel{this deleted text will not be shown}
```
If you want to generate a colored diff automatically between two files, use:
```latex
latexdiff old.tex new.tex > diff.tex
pdflatex diff.tex
```

---
