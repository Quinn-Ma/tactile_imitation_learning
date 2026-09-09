# Manuscript and arXiv submission source

**Compact Visuotactile World Models for Lifting: Prediction, Reward Alignment, and Force Constraints**

Qinzhen Ma (Rice University), Sida Peng (Zhejiang University). Correspondence: qm18@rice.edu.

- [`preprint.pdf`](preprint.pdf): the complete 8-page manuscript, including implementation details and references.
- [`main.tex`](main.tex), [`ieeeconf.cls`](ieeeconf.cls), and the two PNG figures: source files required for compilation.
- [`arxiv_source.zip`](arxiv_source.zip): the source-upload archive, including the frozen code/results package in `anc/reproducibility.zip`.
- [`metadata.txt`](metadata.txt): title, author list, abstract, and comments for the submission form. Empty publication fields do not represent missing research data.
- [`submission_manifest.json`](submission_manifest.json) and [`validation.json`](validation.json): source/archive hashes and local build evidence.

The source archive was extracted into an empty directory and compiled twice with pdfLaTeX / TeX Live 2025, with shell escape disabled. All fonts are embedded; there were no overfull boxes, missing characters, or unresolved citations/references. All eight pages were visually reviewed. Local checks do not establish arXiv-server compilation, public announcement, or conference acceptance. No public identifier has been recorded in this repository.

To compile with an existing TeX Live installation, run from this directory:

```text
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
```

For arXiv, upload `arxiv_source.zip` and select `main.tex` with pdfLaTeX. The generated PDF is provided for reading and comparison. Follow the [official source-submission instructions](https://info.arxiv.org/help/submit_tex.html).

The manuscript discloses AI assistance and distinguishes public sensing records from simulator control. Its publication license is selected separately by the authors; the software license does not relicense third-party material.
