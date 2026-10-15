# Data guide

Original numerical values and file bytes are preserved. `.gz` files use lossless gzip compression. `manifest.json` maps every original path to its public location, schema and SHA256; exact duplicate files are stored once. `file_index.csv` is a spreadsheet-readable index.

| Folder | Contents |
|---|---|
| `reference/` | 50,000 compositions, 550,000 attempted equilibrium states, raw phase instances, conserved elemental partition vectors and failure records. |
| `iid/` | Candidate descriptors, selected features, composition splits, tuning records, IID prediction/reference pairs and final estimators. |
| `composition_ood/` | Extended-composition-domain diagnostic, distinct from holding out an element. |
| `element_ood/` | Six held-out-element tasks with per-state predictions, true labels, fold-specific features and selected model settings. |
| `topology_diagnostics/` | Error decomposition by dominant receiving family, distance diagnostics and historical direct MLP comparisons. |
| `pipeline_selection/` | Historical candidate-pipeline and oracle comparisons, with exact saved predictions. |
| `augmentation_pilot/` | Earlier source-element pilot compositions and cached states reused by the full augmentation stage. |
| `augmentation/` | Five additional source identities for the SECONDARY model; raw/processed results and fixed production feature construction. |
| `production_freeze/` | PRIMARY/SECONDARY checkpoints, scalers, feature schema, split IDs and anchor-sequence freezes before Ni reveal. |
| `ni_predictions/` | Ni descriptor diagnostics and PRIMARY/SECONDARY predictions fixed before reference calculation. |
| `ni_reference/` | 22,000 Ni reference states, raw phases, target vectors, failure/solver records and blind-test scoring. |
| `ni_recovery/` | Anchor sequences and fixed evaluation population; model refits, R0/R1/R2 controls, per-state predictions and checkpoint/history files. |
| `evidence_summary/` | Frozen numerical summaries and original software/environment record. |
| `screening/` | 120,000 candidate compositions, 480,000 surrogate states, applicability filters, Pareto outputs and direct confirmation. |
| `literature_initial/` | Earlier literature extraction/comparison records needed to trace final source decisions. |
| `literature_provenance/` | Provenance repairs, inclusion decisions, source IDs and conversion records. |
| `literature_validation/` | Final matching rules, 149 archived literature conditions, 120 selected comparisons, phase compositions and error grades. |
| `supplementary/` | Thirty-one numbered tables, field dictionary and captions. Begin here for readable summaries. |
| `figure_source_data/` | Numerical panel tables for manuscript Figs. 2–9, descriptor index and stable finalist mapping. |
| `sampling_history/` | Earlier composition lists used only to verify sampling continuation and nonduplication. |

Units: temperatures labelled `_C` are degrees Celsius; `_K` are kelvin; `_wt_pct` denotes mass percent; `_at_pct` denotes atomic percent. Partition components are fractions on [0,1]; phase volume fractions and conserved elemental partition vectors are different quantities. Missing and inapplicable values stay missing. Family order is FCC, BCC, C15, OTHER_SOLID, LIQUID.

Primary keys are `composition_id` for compositions and `state_id` for composition–temperature states. Join features, reference labels and predictions by IDs, never row position. Experimental comparisons use `condition_id` and phase-instance identifiers. Screening uses stable `candidate_id`, not display labels alone.
