# REPRODUCIBILITY_README.md

Generated 2026-09-07T07:17:05+00:00. This package contains no new computation: it consolidates frozen
artifacts. Reproducing the science means re-running the stages below in order, not
re-running STEP5J.

## Software

| component | version |
|---|---|
| thermodynamics | Thermo-Calc 2025b, TC-Python 2025.2.30 |
| database | TCCU6 (Cu-Alloys v6.2) |
| conditions | 1 bar, 1 mol, wt.%, Cu = balance |
| catboost | 1.2.10 |
| matminer | 0.10.1 |
| matplotlib | 3.10.9 |
| mendeleev | 1.2.0 |
| numpy | 2.4.6 |
| optuna | 4.9.0 |
| pandas | 2.3.3 |
| pyarrow | 25.0.1 |
| pymatgen | 2026.5.4 |
| python | 3.14.5 |
| scikit-learn | 1.8.0 |
| scipy | 1.17.1 |
| shap | 0.52.0 |
| torch | 2.12.0+cu132 |
| xgboost | 3.2.0 |

## Execution order

| stage | what it froze |
|---|---|
| FULL_V1 | 550,000 CALPHAD states across six X identities plus baseline |
| STEP5A | 43-feature schema, composition-level splits, IID models, metric definitions |
| STEP5B | composition-domain (Domain B) OOD, opened once under a preregistration |
| STEP5C | strict retrospective leave-one-element-out, six folds |
| STEP5D | LOEO failure diagnosis |
| STEP5E | selection of the zero-shot pipeline family among six candidates |
| STEP5F | five additional chemical identities and the augmented LOEO test |
| STEP5G | production PRIMARY/SECONDARY freeze; Ni compositions and metrics preregistered while Ni was still sealed |
| STEP5H-A | first Ni descriptor access; Ni features and predictions frozen and hashed BEFORE any Ni CALPHAD |
| STEP5H-B | Ni CALPHAD reveal and scoring |
| STEP5I | post-blind few-shot recovery with anchor-only controls |
| STEP5J | this evidence freeze |

The ordering constraint that matters: STEP5H-A predictions were hashed before STEP5H-B
computed a single Ni equilibrium. That hash is what makes the blind test verifiable
rather than merely asserted.

## Seeds and split rules

| item | value |
|---|---|
| STEP5A master seed | 20260904 |
| production master seed | 20260911 |
| PRIMARY child seed | 2266358626 |
| Ni composition sampler | Sobol(d=3, scramble=True), master seed 20260911 |
| few-shot master seed | 20260912 |
| few-shot calibration/evaluation split | SHA256("NI" \| composition_id \| "20260912"), ascending; lowest 30% -> calibration |
| anchor sequences | three nested 32-composition permutations, frozen before the reveal |
| bootstrap | composition-level cluster bootstrap, 1000 replicates |
| post-blind exploratory bootstrap seed | 20260913 |

## Feature schema

43 features, frozen at STEP5A, sha256 `f4ddf4a2587e33a7d441656110c589208d9d860095198dd4b276df64f8a2fb0c`. No re-selection was performed
at any later stage. Temperature_C is both a bookkeeping column and one of the features;
deriving the feature list by excluding metadata columns silently drops it and yields 42.

## Model architecture and metrics

- Architecture: Linear(43,128)-ReLU-Linear(128,64)-ReLU-Linear(64,5)
- Loss: mean squared error on the raw 5 components; optimizer Adam
- Batch 1024, at most 200 epochs, patience 15 on raw validation MSE
- Projection: clip at 0, renormalize; a zero-sum row falls back to the training-mean simplex
- TVD = 0.5 * sum_a |P_pred^a - P_true^a| over the family order FCC, BCC, C15, OTHER_SOLID, LIQUID
- Topology accuracy: fraction of states whose argmax family is correct

## Phase mapping and partition definition

- FCC_L12* -> FCC, BCC_B2* -> BCC, C15_LAVES* -> C15, LIQUID* -> LIQUID, every other
  stable solid -> OTHER_SOLID. Presence threshold: aggregated volume fraction > 1e-4.
- NBNI3, which appeared only in the Ni blind set, was mapped to OTHER_SOLID by this
  frozen default. No new phase family was created after the reveal.
- Partition: `n_X^a = phase_moles_a * X_X^a`; `P_X^a = n_X^a / sum_b n_X^b`.
  Volume-fraction partitioning is forbidden and is not used anywhere.

## Folder structure of this package

| folder | contents |
|---|---|
| 00_freeze | source inventory, freeze verification, FINAL_NUMERICAL_RESULTS.json |
| 01_final_tables | Tables 1-8 plus the three named tables, CSV and Markdown |
| 02_final_figures | Figures 1-12 as PNG 600 dpi, PDF, SVG and source CSV |
| 03_results_draft | JMST_RESULTS_DRAFT.md |
| 04_discussion_draft | JMST_DISCUSSION_DRAFT.md |
| 05_literature_positioning | JMST_LITERATURE_POSITIONING.md |
| 06_claim_audit | FINAL_CLAIM_AUDIT.csv/.md, claim hierarchy |
| 07_provenance | FINAL_PROVENANCE_MAP.csv |
| 08_reproducibility | this file |
| 09_qa | freeze checks, drift report, integrity checks |
| 10_manifest | STEP5J_FILE_HASHES.csv, completion record |
| code_snapshot | the STEP5J modules themselves |

## What this package does not contain

- **dft_or_first_principles_descriptors** - out of scope by design; CALPHAD plus classical elemental descriptors only
- **experimental_validation** - no experimental measurement exists in this project; CALPHAD equilibrium is the only ground truth
- **kinetics** - all labels are equilibrium; no kinetic calculation was performed
- **low_order_binary_ni_anchors** - Cu-Ni / Cr-Ni / Nb-Ni were never computed (STEP5I spec 47)
- **second_prospective_unseen_element** - only Ni was held out prospectively; a second blind element would be a new experiment
