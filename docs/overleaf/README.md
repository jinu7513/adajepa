# Track A research documents / 연구 문서

- `tracka_en.tex`: full English method and experiment protocol.
- `tracka_ko.tex`: corresponding Korean document with the same mathematical definitions.

Both files are standalone: no external figures, bibliography database, or custom style
file is required. The pipeline figure is drawn in TikZ. References are included in
the source. Neither document contains fabricated benchmark results.

## Overleaf

1. Create a blank project and upload the `.tex` file (or upload the supplied ZIP).
2. In **Settings / Compiler**, choose **XeLaTeX**.
3. Select `tracka_en.tex` or `tracka_ko.tex` as the **Main document**.
4. Recompile. Compile again if cross-references or the table of contents need updating.

한국어: 새 프로젝트에 원하는 `.tex` 파일을 올리고 컴파일러를 **XeLaTeX**로 선택합니다.
영문/국문을 함께 올렸다면 **Main document**에서 원하는 파일을 지정합니다.
국문은 TeX Live의 `kotex`, `UnBatang`, `UnDotum`을 사용하며, 해당 폰트가 없는
Windows 로컬 환경에서는 설치된 Malgun Gothic을 fallback으로 사용합니다.

## Contents / 수록 내용

Research question and motivation; hypothesis versus established evidence; observation
model; canonical-state reset checks; color/corruption distributions; dataset sampling
and trajectory splits; ViT/MAE tensor shapes; additive and cross-attention conditioning;
exact clean/render/corruption/global losses; alternating algorithm and compute budget;
frozen ridge probes and metrics; default/red/blur/severity-grid evaluation; proposed
12-run primary matrix and ablations; statistical interpretation; W&B and recovery;
checkpoint/resume; limitations; actual runnable commands and code map.

## Evidence boundary

Implementation/config snapshot: 2026-09-27, branch `robust-encoder-scratch`, reviewed
base `66c5f18d00cc627ae35b9b183367f7e510972106`. Actual source hashes are written to
experiment metadata. The 24-test implementation validation is documented separately
in `../tracka_validation.md`. Planned multi-seed experiments have not been run.

The English/Korean documents describe the same implementation. A few explanatory
phrases are localized rather than translated word-for-word. English variable names,
config keys, mathematical notation, and executable commands are kept consistent.

## Local PDF validation

Both documents were compiled with Tectonic 0.17.0 (XeTeX engine), and all pages
were rendered and visually inspected. English: 16 pages; Korean: 17 pages with
the local Korean-font fallback. Overleaf font availability may change pagination.
The supplied PDFs are in `../../output/pdf/`. The source ZIP contains both standalone
TeX files and this README; select the desired main document after uploading it.
