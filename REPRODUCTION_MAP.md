# Manuscript-to-repository map

| Manuscript component | Readable data entry | Detailed data and source implementation |
|---|---|---|
| Reference population and phase-family definitions, Fig. 2 | Supplementary Data S1–S5, S30–S31; `figure_source_data/frozen_panel_tables/FIG2*` | `data/reference/`; `src/stages/reference/code_snapshot/` |
| Descriptors, feature selection and composition splits | Supplementary Data S4–S5 and S26; `figure_source_data/frozen_panel_tables/DESCRIPTOR_INDEX_43.csv` | `data/iid/02_features/`, `data/iid/03_splits/`; IID source scripts 31–39 |
| IID benchmark, Fig. 3 | Supplementary Data S6; `figure_source_data/frozen_panel_tables/FIG3*` | `data/iid/05_results/`; IID source scripts 40–43; `reproduce.py summary` |
| Six element-holdout tests, Fig. 4 | Supplementary Data S7; `figure_source_data/frozen_panel_tables/FIG4*` | `data/element_ood/05_predictions/`; element-OOD source; `reproduce.py summary` |
| Dominant-family diagnostics and oracle comparisons, Fig. 5 | Supplementary Data S8–S10; `figure_source_data/frozen_panel_tables/FIG5*` | `data/topology_diagnostics/`, `data/pipeline_selection/`; matching source directories |
| Prospective Ni predictions and descriptor coverage, Fig. 6 | Supplementary Data S11; `figure_source_data/frozen_panel_tables/FIG6*` | `data/production_freeze/`, `data/ni_predictions/`, `data/ni_reference/`; `reproduce.py inference` and `summary` |
| Ni recovery, anchor controls, Fig. 7 | Supplementary Data S12–S16; `figure_source_data/frozen_panel_tables/FIG7*` | `data/ni_recovery/`; source scripts 111–115; `reproduce.py summary`; original `113_i_metrics.py` |
| Literature comparisons, Table 2 and Fig. 8 | Supplementary Data S17–S21, S27–S28; `figure_source_data/frozen_panel_tables/FIG8*` | `data/literature_validation/` and provenance predecessors; `src/stages/literature_validation/code_snapshot/` |
| Candidate screening, Table 3 and Fig. 9 | Supplementary Data S22–S25 and S29; `figure_source_data/TABLE3_VERIFIED_ID_MAP.csv`, `figure_source_data/frozen_panel_tables/FIG9*` | `data/screening/04_virtual_screening/` and confirmation outputs; screening source scripts 130 onward |
| SI data preparation | `data/supplementary/` | `src/stages/supplementary/` |

File patterns may resolve to a gzip-compressed file or to a manifest alias. Use `data/file_index.csv` to find the physical storage path. The execution workspace restored by `prepare` recreates every original path.

The summary/inference commands regenerate numerical results. Archived analytical plotting scripts document the scientific plotting logic. Later cosmetic rearrangement, schematic artwork, Word assembly and author-side layout QA are separate from these numerical checks; the final numerical panel tables are supplied for independent replotting.

The original source snapshots are preserved for inspection. Not all historical bookkeeping programs are standalone reproduction steps: initial task-attachment capture, publisher-fulltext retrieval and immutable original-run hash audits belong to the original provenance record. Follow the documented numerical commands first.
