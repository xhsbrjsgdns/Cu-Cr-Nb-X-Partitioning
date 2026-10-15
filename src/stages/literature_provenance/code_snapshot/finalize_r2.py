from pathlib import Path
import json, hashlib, csv
import pandas as pd

O=Path(__file__).resolve().parents[1]
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,s): (O/p).write_text(s,encoding='utf-8')
def table(p):return pd.read_csv(O/p,keep_default_na=False)
baseline=json.loads((O/'00_config/protected_before.json').read_text(encoding='utf-8'))
checks=[dict(path=r['path'],before_sha256=r['sha256'],after_sha256=sha(r['path']) if Path(r['path']).exists() else 'MISSING') for r in baseline]
for r in checks:r['unchanged']=r['before_sha256']==r['after_sha256']
pd.DataFrame(checks).to_csv(O/'12_qa/PROTECTED_HASH_VERIFICATION.csv',index=False)
assert all(r['unchanged'] for r in checks), 'Protected file changed'
inv=table('01_source_inventory/STEP5K_R2_LOCAL_SOURCE_INVENTORY.csv')
assert all(sha(r.full_path)==r.sha256 for r in inv.itertuples()), 'R1 inventory changed'
ev=table('04_fulltext_evidence/STEP5K_R2_FULLTEXT_EVIDENCE.csv')
v=table('07_external_validation/STEP5K_R2_EXTERNAL_VALIDATION.csv')
old=table('08_old_calphad_audit/STEP5K_R2_OLD_CALPHAD_AUDIT.csv')
states=table('09_optional_new_nonNi_calphad/STATES.csv')
assert len(ev)==len(v)==42 and v.validation_id.is_unique
assert len(old)==12 and old.converged.astype(str).str.lower().eq('true').sum()==4
assert v.agreement_class.value_counts().to_dict()=={'NOT_COMPARABLE':40,'PARTIALLY_CONSISTENT':2}
assert len(states)==3 and states.converged.astype(str).str.lower().eq('true').sum()==0
assert set(v[v.agreement_class=='PARTIALLY_CONSISTENT'].X_identity)=={'MO'}
s=json.loads((O/'STEP5K_R2_SUMMARY.json').read_text(encoding='utf-8'))
s.update(protected_files_verified_unchanged=len(checks),protected_hash_changes=0,R1_inventory_files_verified_unchanged=len(inv),Ni_files_modified=False,production_model_hash_changed=False,direct_successful_CALPHAD_pairs=0)
(O/'STEP5K_R2_SUMMARY.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
write('12_qa/STEP5K_R2_INTEGRITY_AUDIT.md',f'''# STEP5K-R2 integrity audit

All {len(checks)} files in the pre-execution protected baseline were reread and SHA-256 verified unchanged. This baseline covers STEP5G, STEP5H, STEP5I, STEP5J, virtual screening and screening confirmation, including the production models and scalers. Individual before/after hashes are in PROTECTED_HASH_VERIFICATION.csv. All {len(inv)} inventoried R1 source/workflow files also retain their inventoried SHA-256 hashes.

No Ni calculation, prediction, composition modification, model retraining, scaler refit, feature reselection, hyperparameter search, new prospective element, DFT, finalist replacement or screening modification was executed. Reading protected files for verification is not a modification. The audit establishes byte identity for the enumerated baseline, not a system-wide filesystem activity trace.

The only new thermodynamic requests were three non-Ni Si states at two source compositions. All failed. R2 outputs are separate from frozen evidence. No old metric was recomputed or overwritten.

The final manifest covers delivery artifacts other than itself and transient Python bytecode. Its own hash cannot be included recursively. Verification script: code_snapshot/finalize_r2.py.
''')
write('12_qa/STEP5K_R2_QA.md','''# Delivery QA

PASS: 42 unique condition IDs link evidence and validation rows; 12 old states audited, including four converged states; two retained qualitative Mo comparisons; 40 NOT_COMPARABLE rows; three new states, zero converged. No successful direct validation pair is claimed. All 471 protected hashes and 46 inventoried R1 hashes agree with their saved baselines.

Source identity, condition completeness, matching eligibility and agreement are separate fields. DIRECTLY_COMPARABLE describes a source condition and does not establish a successful computed pair. Failed equilibrium attempts do not establish absence of phases. Duplicate PDF copies are not additional papers. Unknown composition basis, trace constituents and contradictory nominal compositions are not silently corrected. Literature descriptions after cooling or deformation do not establish equilibrium at the annealing or sintering temperature.

The fixed atomic-weight conversion convention uses Zr=91.224; all admitted Zr compositions are already wt%, so this value affects none of the delivered conversions. No normalization removes reported trace constituents. R2 conclusions concern the specified target topology and are deliberately narrower than general bibliographic relevance.
''')
answers=[
('Existing local sources verified','8 unique documents (7 journal papers + 1 thesis), represented by 12 PDF copies.'),
('Old source mappings correct','0 of the 8 unique-document mappings.'),('Mismatched mappings','8; corrected from actual document identity.'),('Ambiguous source identities','0. This does not remove ambiguity within experimental conditions.'),
('New candidates found','6 genuinely new candidates; 9 records reviewed including 3 rediscovered R1 candidates.'),('New legal full texts','3 genuinely new full texts. Two additional R1 candidate full texts were recovered.'),('New usable papers','3, for qualitative experimental context.'),('User PDF requests','10 high-value papers remain on the request list.'),('Verified primary papers in total','12 journal papers plus 1 primary thesis; 9 journal papers supply at least one admitted usable condition.'),
('DIRECTLY_COMPARABLE conditions','13 source conditions; zero successful direct CALPHAD pairs.'),('QUALITATIVELY_COMPARABLE conditions','15.'),('NOT_DIRECTLY_COMPARABLE conditions','14.'),('CONSISTENT comparisons','0.'),('PARTIALLY_CONSISTENT comparisons','2 Mo comparisons, qualitative only.'),('DISCREPANT_BUT_EXPLAINABLE comparisons','0. Unmatched or failed calculations are NOT_COMPARABLE, not discrepancies.'),
('Zr evidence','LIMITED: Cr precipitation, reported Cu3Zr and Zr segregation; requested Zr-Laves topology not verified.'),('Ti evidence','LIMITED: Cr/Ti enrichment; no admitted confirmation of Cu4Ti topology.'),('Co evidence','LIMITED: sigma/Cr/Cu in a Cr-rich thesis alloy; Co-Nb Laves unsupported.'),('Si evidence','LIMITED: Cu-Nb-Si equilibrium observations; no Cr3Si competition evidence and all new calculations failed.'),('Ag evidence','LIMITED: Cr precipitation and processed filaments; no admitted Cu-rich/Ag-rich FCC splitting confirmation.'),('Mo evidence','LIMITED: refractory Cr/Mo chemistry; no two-BCC topology or phase-fraction validation.'),
('Exact FULL_V1 composition + temperature matches','0.'),('Tight neighbors','0.'),('Distant neighbors removed','All 14 old nearest-neighbor validation rows removed. R2 separately classifies 12 of 42 conditions as DISTANT_NEIGHBOR and 30 as NO_MATCH; none is admitted as validation.'),('Old four converged states retained','2 retained only as qualitative Mo context; 0 retained as direct equilibrium validation.'),('Incorrect metadata in previous 12 states','Yes. Document associations and several composition interpretations required repair; the complete 12-row audit separates source, composition and temperature errors.'),('New non-Ni CALPHAD necessary','Justified for selected source-verified equilibrium conditions lacking frozen matches. Only 2 Si compositions / 3 states were attempted.'),('New converged states','0 of 3; all failed with QTHISS TOO MANY ITERATIONS. No phase outcome inferred.'),('Ni files touched','No modifications; no Ni calculations or predictions. Protected baseline hashes unchanged.'),('Models retrained','No.'),('Production-model hashes changed','No; all 471 protected baseline files verified unchanged.'),('Safe for JMST use','Yes as narrowly qualified external experimental context and two qualitative Mo comparisons. No broad validation or numerical-accuracy claim.'),('Ready for final assembly','No. Source provenance is repaired, but requested topology validation and baseline Cu-Cr-Nb evidence remain incomplete.')]
qa='\n'.join(f'| {i} | {q} | {a} |' for i,(q,a) in enumerate(answers,1))
write('STEP5K_R2_REPORT.md','''# STEP5K-R2 source provenance repair — completed within scope

R2 repairs the external-evidence chain but does **not** establish broad experimental validation of the frozen model. Eight existing document identities were verified and all eight old summary associations corrected. The old DOIs are real bibliographic records; the fault was their association with different local PDFs. Duplicate copies were deduplicated. The original R1 and frozen artifacts were preserved.

The reconstructed set contains 42 conditions from 12 verified journal papers and one thesis. Thirteen conditions have sufficiently complete source equilibrium information to be classified DIRECTLY_COMPARABLE, but there are **zero successful direct computed pairs**. Only two old Mo states remain as qualitative, partially consistent refractory phase-chemistry context. No validation accuracy is calculated.

## Evidence decisions

Actual full-text title/author/year/DOI identity was checked against publisher or DOI metadata. Twenty old DOI records were checked separately; this does not imply that twenty corresponding full texts were available. Page/table/figure pointers, composition basis, heat treatment, actual phase observations and exclusions are retained in the source and evidence tables.

Four contradictory Cu-Nb-Si table/narrative compositions were excluded without inferring corrections. Six historical Cu-Cr-Ti conditions have unresolved composition basis. Reported trace elements remain explicit. A Cr3Ti assumption used for volume-fraction estimation is not treated as phase identification. Cu-Nb-Si observations do not establish Cr3Si; Cu-Cr controls do not establish Cu-Cr-Nb baseline behavior.

Composition conversion and matching rules were frozen before judging agreement. All 45,000 FULL_V1 compositions were considered under conservative within-element geometric criteria and exact temperature requirements. No exact or tight composition-temperature match was found. Distant neighbors provide no external validation evidence.

The two retained Mo cases compare hot calculated BCC + LIQUID with experimentally characterized Cu + Mo-Cr after cooling. They support only qualitative refractory Cr/Mo chemistry, not identical phase assemblages, phase fractions or BCC/BCC splitting. Three new source-specific Si equilibrium requests failed; these failures remain in the canonical parquet and are not replaced by inferred phases.

## Answers to all required questions

| No. | Question | Answer |
|---|---|---|
'''+qa+'''

## Delivery and remaining gaps

- [External validation interpretation](07_external_validation/STEP5K_R2_EXTERNAL_VALIDATION_REPORT.md)
- [Source provenance map](02_pdf_provenance/STEP5K_R2_SOURCE_PROVENANCE_MAP.csv)
- [Condition-level evidence](04_fulltext_evidence/STEP5K_R2_FULLTEXT_EVIDENCE.csv)
- [Element coverage](07_external_validation/STEP5K_R2_ELEMENT_COVERAGE.csv)
- [High-value PDF request list](03_new_literature_search/STEP5K_R2_HIGH_VALUE_PDF_REQUEST_LIST.csv)
- [Integrity audit](12_qa/STEP5K_R2_INTEGRITY_AUDIT.md)

Four replacement JMST insertion files are in 10_manuscript_insertions. They are review-ready prose, not an assertion that the full manuscript is ready. Additional phase-equilibrium full texts and verified Cu-Cr-Nb baseline evidence remain needed. No literature claim is based solely on an abstract. No statistical independence from the thermodynamic database's underlying assessments is asserted.

**STOP:** R2 is complete within the authorized repair scope. No Ni work, second prospective element, optimization, retraining, DFT, screening changes or finalist changes follow. Await user review.
''')
write('07_external_validation/STEP5K_R2_EXTERNAL_VALIDATION_REPORT.md','''# Reconstructed external validation

The evidence supports narrow qualitative context, not broad model validation. Source-condition comparability and computed agreement must not be conflated: 13 direct, 15 qualitative and 14 excluded conditions yield zero successful direct pairs, two PARTIALLY_CONSISTENT qualitative Mo pairs and 40 NOT_COMPARABLE rows.

The retained Mo states originate from the additive-free Cu-10Cr-30Mo (wt%) controls reported by [Cho et al., Metals 2021](https://doi.org/10.3390/met11050700) and [Cho et al., Archives of Metallurgy and Materials 2021](https://doi.org/10.24425/amm.2021.136361). Post-cooling characterization establishes refractory Mo-Cr chemistry. Calculated BCC + LIQUID at 1200 or 1250 °C is not an experimental confirmation of an identical hot assemblage. Oxide-added specimens are not reduced to ternaries by deleting oxides.

[Zhou et al.](https://doi.org/10.1016/S1003-6326(23)66149-7) provide Cu-Nb-Si long-anneal evidence. Twelve source conditions are internally unambiguous; four composition contradictions are quarantined. Three selected states at two compositions were attempted and failed with QTHISS TOO MANY ITERATIONS. They supply neither phase-presence nor phase-absence evidence.

New qualitative sources include [Sumino et al.](https://doi.org/10.34562/jic.58.1_64), [Chbihi et al.](https://doi.org/10.1016/j.actamat.2012.01.038) and [Hughes et al.](https://doi.org/10.1016/j.jmrt.2022.07.113). Cr/Ti enrichment, Cr precipitation and Zr segregation are source observations. They do not establish the requested Cu4Ti, Zr-Laves or multi-phase partition topologies. Post-deformation and ageing evidence is explicitly separated from equilibrium evidence. No verified Cu-Cr-Nb baseline condition is admitted.

All 14 old nearest-neighbor rows are withdrawn from validation. The new matching table separately contains 12 distant-neighbor and 30 no-match conditions; exact and tight matches are both zero. The six target-element evidence strengths remain LIMITED, and the Cu-Cr-Nb baseline remains NONE. Agreement percentages would be misleading and are not reported.
''')
insertions={
'METHODS':'''External experimental evidence was audited at the document and condition levels. Each admitted observation was linked to a verified full text, composition basis, thermal history and phase-characterization location. Duplicate documents were not counted independently; unresolved composition bases, contradictory compositions and unreported trace constituents were not silently repaired. Literature conditions were classified separately from calculation agreement. Composition conversion and conservative matching criteria were fixed before evaluating agreement, with exact temperature matching required. Distant neighbors were excluded. Post-freeze non-Ni calculations were used only for external comparison and were not incorporated into training, scaling, model selection or historical OOD evaluation.''',
'RESULTS':'''The audited evidence set comprised 42 conditions from 12 journal papers and one thesis. Thirteen source conditions were classified as directly comparable in principle, 15 as qualitatively comparable and 14 as not directly comparable. No exact or tight FULL_V1 composition–temperature match was identified, and no successful direct equilibrium comparison was established. Two source-verified legacy Cu–Cr–Mo states showed partial qualitative agreement in refractory Cr/Mo phase chemistry. Because the experiments characterized cooled specimens whereas the calculations described the sintering temperature, these comparisons do not validate identical phase assemblages or phase fractions. Three additional Cu–Nb–Si equilibrium calculations failed to converge and supplied no phase evidence.''',
'DISCUSSION':'''The external literature provides limited physical context for the predicted trends. Cr/Ti enrichment and Cr precipitation with Zr segregation are experimentally documented, while Cu–Nb–Si long-anneal observations constrain adjacent silicide chemistry. These observations do not independently verify Cu4Ti, Zr- or Co-containing Laves phases, Cr3Si competition, Cu-rich/Ag-rich FCC separation or two-BCC partition topology across the screened composition space. The retained Mo comparisons support refractory phase chemistry only. Consequently, the present evidence does not justify a general experimental-validation claim or a numerical accuracy estimate. Further composition-specific phase-equilibrium measurements and verified Cu–Cr–Nb baseline evidence are required.''',
'SUPPLEMENTARY':'''The R2 supplementary audit records local-file hashes, repaired document identities, legal full-text access, page/table/figure evidence pointers, nominal and measured compositions, conversion assumptions, matching criteria and condition-level inclusion decisions. The 12 historical external calculations are audited individually: of four converged states, two are retained only for qualitative Mo context. All 14 historical nearest-neighbor validation rows are excluded. Three new failed non-Ni states are retained in the canonical results table with explicit errors; missing phase output is not interpreted as phase absence. The 471 protected baseline files and 46 inventoried R1 files retain their saved SHA-256 hashes. Frozen production models, scalers, Ni blind evidence, screening results and historical metrics were not modified.'''}
for name,body in insertions.items():
    write(f'10_manuscript_insertions/STEP5K_R2_JMST_{name}_INSERTIONS.md',f'# JMST {name.lower()} replacement insertion\n\n'+body+'\n\nEditorial note: replace only the external-literature passage; retain frozen numerical results. Cite the source-specific claims using the verified provenance map and condition evidence table. Do not describe the 13 source conditions as 13 successful validation pairs.\n')
manifest=[]
for p in sorted(O.rglob('*')):
    if p.is_file() and '__pycache__' not in p.parts and p.name!='STEP5K_R2_FILE_HASHES.csv':
        manifest.append(dict(relative_path=p.relative_to(O).as_posix(),size=p.stat().st_size,sha256=sha(p)))
pd.DataFrame(manifest).to_csv(O/'13_manifest/STEP5K_R2_FILE_HASHES.csv',index=False)
print(json.dumps(dict(protected_unchanged=len(checks),R1_unchanged=len(inv),manifest_files=len(manifest),condition_rows=len(v),status='PASS')))
