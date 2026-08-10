# Anonymous ICAIF 2026 source

This directory contains the complete source for the anonymous eight-page
submission.

Compile from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The build uses the bundled official `acmart` 2.19 class and ACM bibliography
style. Rename the resulting `main.pdf` to
`SAFE_ALPHA_ICAIF26_submission.pdf` before upload.

The submission source comprises:

- `main.tex`;
- `body.tex`;
- `references.bib`;
- `figures/calibration_power.pdf`;
- `figures/evidence_path_repaired.pdf`;
- `acmart.cls`; and
- `ACM-Reference-Format.bst`.

ICAIF 2026 permits at most eight total pages, including references and
figures, and does not accept supplementary material or appendices. Do not add
the separate extended proof manuscript to the CMT submission.
