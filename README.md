# Cu–Cr–Nb–X solute partitioning: data and code

Companion materials for **Cross-Element Generalization and Recovery of Solute Partitioning Predictions in Cu–Cr–Nb–X Alloys**.

This repository preserves the numerical inputs, reference labels, prediction outputs, fitted production models, frozen protocols and original scientific source scripts underlying the submitted manuscript. It covers the original Cu-based study, prospective Ni test, post-blind recovery, literature comparisons and candidate screening. Later experiments in the local `10_JCTC_UPGRADE` project are outside this release.

## Start here

- For a readable account of the data, open [data/README.md](data/README.md).
- For the paper and SI tables, start with [data/supplementary/tables](data/supplementary/tables) and [the field dictionary](data/supplementary/Supplementary_Data_Dictionary.csv).
- For numerical figure-panel data, open [data/figure_source_data](data/figure_source_data).
- For the original scientific implementation, open [src/stages](src/stages), following the stage order below.
- [REPRODUCTION_MAP.md](REPRODUCTION_MAP.md) links manuscript figures and tables to the data and corresponding source stages.
- [data/file_index.csv](data/file_index.csv) is the complete file index. [data/manifest.json](data/manifest.json) maps original paths to public storage paths, checksums and table schemas.

```text
data/
  reference/                # compositions, equilibrium outputs, labels, failures
  iid/                      # descriptors, splits, tuning, IID results and models
  element_ood/              # six leave-one-element-out tasks
  topology_diagnostics/     # dominant-family errors and related diagnostics
  production_freeze/        # frozen PRIMARY/SECONDARY models and protocols
  ni_predictions/           # descriptor coverage and predictions before reveal
  ni_reference/             # Ni equilibrium labels and blind-test results
  ni_recovery/              # anchor sequences, refits and anchor-only controls
  literature_validation/    # final experimental-condition comparisons
  screening/                # candidate population, filters, Pareto and confirmation
  supplementary/            # 31 reader-facing tables and field dictionary
  figure_source_data/       # numerical sources for manuscript panels
  ...                       # source augmentation and provenance dependencies
src/
  reproduce.py              # portable verification, summary, inference and launcher
  stages/                   # original implementation grouped by scientific stage
```

The additional folders retain the dependencies of the SECONDARY model, the composition-domain diagnostic, earlier literature extraction and sampling-continuation checks. They are identified individually in the data guide.

## Quick numerical reproduction

Run from this repository root. Python 3.14 was used in the archived workflow; the summary checker needs only NumPy, pandas and PyArrow. The observed checks in this release are recorded in [VALIDATION_REPORT.json](VALIDATION_REPORT.json).

```bash
python -m pip install -r requirements.txt
python src/reproduce.py verify
python src/reproduce.py summary --output reproduced_results
```

`verify` checks all data and source checksums and compiles the archived Python sources without executing them. `summary` recalculates the reference-state counts, IID F1/TVD, six-fold LOEO TVD, pooled topology-error share, Ni blind TVD and topology accuracy, three-replicate recovery means, literature grades and screening counts. It uses released state-level reference/prediction arrays; it does not merely copy manuscript values.

The expected main values are IID X-partition TVD **0.0016401332**, unweighted six-fold LOEO TVD **0.8504979259**, Ni blind TVD **0.8643856187**, and refit TVDs **0.2481623701 / 0.0507380675 / 0.0271879848** at 1 / 8 / 32 composition anchors. Each Ni anchor supplies eleven temperature states. Summary files and recomputed CSVs are written to the selected output directory.

## Rerun frozen-model predictions

```bash
python -m pip install -r requirements-models.txt
python src/reproduce.py inference --output reproduced_results
```

This reruns PRIMARY and SECONDARY inference on all 22,000 Ni states using their released checkpoints, frozen feature order, scalers, exact network architecture and simplex projection. The release check produced zero maximum difference from both archived prediction arrays on the local CPU environment. It does not retrain either model.

## Inspect or rerun original stages

The original source files remain separate and retain their original filenames. The archived workflow used fixed Windows paths and some path-bearing CSV/JSON ledgers. Preparation restores a separate working tree and substitutes the old project root in source and text metadata; it records these changes in `RELEASE_PORTABILITY.json`. The immutable release is left untouched. Numerical arrays and original frozen records remain available with their original checksums.

```bash
python src/reproduce.py list-stages
python src/reproduce.py prepare --workspace ../CuCrNbX_work
python src/reproduce.py run --workspace ../CuCrNbX_work 21_fullv1_compositions.py -- --verify
python src/reproduce.py run --workspace ../CuCrNbX_work 113_i_metrics.py
```

Put `--workspace` before the script name. Pass any script arguments after `--`. Select the Python environment appropriate to the stage. `prepare` requires an empty workspace outside this repository and several GB of free disk space. Original stages may overwrite outputs **inside that prepared workspace**. Historical audit scripts that compare source or path-bearing metadata hashes are records of the original execution and are not expected to validate byte-for-byte after path substitution. Do not disable or silently update their historical hash gates.

Scientific stage order is reference → IID → composition OOD / element OOD → topology diagnostics → pipeline selection and source augmentation → production freeze → Ni predictions → Ni reference reveal → Ni recovery → evidence/screening/literature comparison. Do not treat reproducing the historical ordering today as a new prospective test. Some original preparation/audit scripts refer to author-side task attachments or local plotting-layout QA tools; those administrative tools are not required by the tested summary and inference commands. The archived stage plotting scripts and released panel tables document the analytical figures; final Word composition and local presentation-layout tooling are outside these numerical reproduction commands.

The large optional LOEO/oracle/augmentation estimator caches are omitted; their training code, settings, source data and saved predictions are retained. Exact omissions are listed in `excluded_model_caches` in the manifest. IID final estimators and the PRIMARY/SECONDARY/recovery checkpoints are retained.

## File formats and scientific interpretation

- `composition_id` identifies a composition; `state_id` identifies a composition–temperature state. Join by these IDs, not row position.
- Partitions are mole-balanced elemental inventories over `FCC, BCC, C15, OTHER_SOLID, LIQUID`, in that order. They are not phase volume fractions.
- The reference space has 550,000 attempted states, 548,432 converged states and 1,568 Mo failures at 950 °C. Missing reference values remain missing; no failed labels were imputed.
- The 22,000-state Ni blind population is distinct from the 15,400-state recovery evaluation population. All anchor strategies share the latter evaluation set.
- LOEO manuscript averages are unweighted means of six element-fold means. The 96.7% topology-error share is pooled error mass, not a mean of fold percentages.
- The final literature comparison uses 120 selected conditions from 149 archived records; the latter must not be counted as 149 successful direct validations.
- The five global finalists were selected with surrogate objectives. Their direct confirmation does not establish a new global ranking of the full candidate population.

Tables larger than 8 MiB are generally stored with lossless gzip compression. A few large compressed files are divided into consecutive `.part001`, `.part002`, … pieces to remain below GitHub's ordinary Git file-size limit. `source_stream`, `summary`, `inference` and `prepare` handle these transparently. Pieces are not independent tables. Duplicate files are stored once and mapped to all their original locations by the manifest. To reconstruct a single original file manually, concatenate its pieces in manifest order and then decompress gzip. Ordinary `.parquet` and `.csv` files open directly; a `.parquet.gz` must first be decompressed before use with tools that do not accept gzip streams.

## Environment and thermodynamic software

The complete original software/version record is retained in [ENVIRONMENT_ORIGINAL.md](ENVIRONMENT_ORIGINAL.md). The minimal and model requirements are separated so numerical checks do not require a thermodynamic installation or all descriptor-generation libraries.

`requirements-training.txt` lists additional descriptor-generation, tuning and plotting dependencies. Historical stages were developed on Windows; the source-path restoration has been checked on Windows. The direct `summary` and `inference` commands do not use those historical absolute paths.

Original equilibria used Thermo-Calc 2025b, TC-Python 2025.2.30 and TCCU6 (Cu-Alloys v6.2). Thermo-Calc and its commercial database are not distributed here. Fresh equilibrium calculations require the appropriate licensed software/database and a matching TC-Python environment. The deposited equilibrium outputs support ML/statistical reproduction without access to the commercial database. Descriptor regeneration additionally uses the packages and elemental-property versions in the original environment record; the actually used descriptor values are already deposited.

## Publication, version and rights

The public companion repository is [https://github.com/xhsbrjsgdns/Cu-Cr-Nb-X-Partitioning](https://github.com/xhsbrjsgdns/Cu-Cr-Nb-X-Partitioning). The publication snapshot is identified by release `v1.0.0`. No DOI has been assigned. No general open-source/data license has been assigned by this upload; third-party rights remain with their owners. No license grant is inferred for Thermo-Calc or its database. Publisher PDFs, editorial correspondence, credentials and license-server settings are not part of this package.

The data are about 2 GB. Download or clone the complete tree and retain every split-file part. Each physical file is below 100 MiB. The repository release provides a fixed version for the manuscript; a later Zenodo archive can add a persistent DOI.
