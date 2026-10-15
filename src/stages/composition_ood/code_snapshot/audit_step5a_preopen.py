from pathlib import Path
import hashlib, json
from datetime import datetime, timezone
import pyarrow.parquet as pq
A=Path('E:/Cu_CALPHAD_ML/04_ML/STEP5A_JMST_ML_FOUNDATION_V2')
O=Path('E:/Cu_CALPHAD_ML/05_OOD/STEP5B_DOMAIN_B_OOD_V1')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
fpath=A/'09_manifest/DEVELOPMENT_FREEZE_V1.json'
f=json.loads(fpath.read_text(encoding='utf-8'))
checks=[]
def check(path, expected):
    actual=sha(path)
    checks.append(dict(path=str(path),expected_sha256=expected,actual_sha256=actual,status='PASS' if actual==expected else 'FAIL'))
for task,t in f['tasks'].items(): check(A/f'04_models/final/{task}/model_state.pkl',t['artifact_sha256'])
for name,key in [('PRIMARY_FEATURE_LIST.txt','feature_list_sha256'),('PCC_REMOVAL_LOG.csv','pcc_removal_sha256'),('NEAR_ZERO_VARIANCE_REMOVAL.csv','nzv_removal_sha256'),('EXACT_REDUNDANCY_LOG.csv','exact_redundancy_sha256')]: check(A/'02_features'/name,f[key])
m=json.loads((A/'09_manifest/CODE_SNAPSHOT_MANIFEST_STEP5A_FINAL.json').read_text())
for item in m['files']: check(A/'code_snapshot'/item['filename'],item['sha256'])
features=(A/'02_features/PRIMARY_FEATURE_LIST.txt').read_text().splitlines()
assert features==f['features'] and len(features)==len(set(features))==43
assert list(f['tasks']['presence']['thresholds'].values())==[.5,.37,.5,.32,.42]
assert all(c['status']=='PASS' for c in checks)
footer_rows=pq.ParquetFile('E:/Cu_CALPHAD_ML/03_full_CALPHAD/FULL_V1/03_processed/FULL_V1_ML_READY_CONVERGED.parquet').metadata.num_rows
now=datetime.now(timezone.utc).isoformat()
evidence=[]
for rel in ['code_snapshot/step5a_common.py','code_snapshot/39_step5a_development_freeze.py','code_snapshot/40_step5a_iid_test.py','08_logs/RERUN_PATCH01_CHAIN.log','09_manifest/STEP5A_PATCH_01.md','09_manifest/STEP5A_PATCH_02.md','05_results/TEST_OPEN_RESUME_RECORD.json','07_qa/LEAKAGE_AUDIT_FINAL.md']:
    evidence.append({'path':str(A/rel),'sha256':sha(A/rel)})
summary=dict(status='FAIL',scope='Pre-opening protocol audit; OOD evaluation NOT EXECUTED',generated_utc=now,development_freeze_sha256=sha(fpath),artifact_checks=checks,feature_names_order_count='PASS: 43',thresholds=f['tasks']['presence']['thresholds'],historical_target_file_footer_rows=footer_rows,footer_only_read=True,seal_audit='FAIL under literal never-read-targets requirement',reason='Frozen STEP5A load_development reads all target rows without a domain filter before a left merge with Domain-A candidate features. The rerun log records execution of this path. A new STEP5B preregistration cannot precede that historical read.',evidence=evidence,current_run_target_rows_read=0,current_run_predictions=0,current_run_metrics=0,preregistration_sha256=None,first_open_timestamp=None,primary_metrics=None,go_no_go='NO-GO: protocol gate',training_leakage='Not demonstrated by this finding; target materialization is distinct from training or tuning use.',anchor_seal='Historical target materialization also covers the shared full database; no Anchor predictions or metrics created in this run.',ni_seal='No descriptor lookup, composition, calculation, or prediction performed in this run.',stages={'A':'Artifact hashes PASS; historical seal inconsistency found','B':'Directory created for audit evidence','C':'NOT EXECUTED','D':'Evaluation snapshot NOT EXECUTED; audit-only script retained','E':'FAIL: strict first-read assertion cannot be certified'},patch01='Duplicate legal feature fix remains distinct from this loader finding; no inference of second test tuning.')
(O/'07_qa/STEP5B_DOMAIN_B_OOD_SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
report='''STEP 5B STATUS: FAIL

# 开封前协议审计：已停止，未执行 OOD 评价

本次核验通过：5 个模型、43-feature list、PCC/NZV/exact-redundancy 文件及最终快照的 17 个源码文件，合计 26 项 SHA256 全部一致。43 个特征的名称、顺序、唯一性与 freeze 一致；presence thresholds 为 0.50/0.37/0.50/0.32/0.42。

## 阻断证据

- `STEP5A/code_snapshot/step5a_common.py:83-87`：使用 `pd.read_parquet(...FULL_V1_ML_READY_CONVERGED.parquet, columns=[state_id]+tgt_cols)`，没有行或 domain 过滤；随后才与 Domain A candidate features 左合并。
- 同文件 90-93 行对完整数据库读取 phase-split diagnostics，随后才合并。
- `39_step5a_development_freeze.py:248-250` 调用该函数，再选择 DEV_TRAIN/DEV_VALIDATION。
- `08_logs/RERUN_PATCH01_CHAIN.log` 记录上述冻结链及 IID resume 完成，CHAIN EXIT: 0；快照哈希也吻合。
- FULL_V1 的 `27_fullv1_postprocess_recovery.py:344-349` 显示源表按 state_converged 筛选，没有 Domain A 限定。本次只读取 parquet footer，确认 548432 行，没有读取 target 数据页。

因此，历史执行曾将完整 target 列物化到内存；现有记录不能支持 STEP5B 所要求的“preregistration 在第一次读取 Domain B targets 之前”。不能事后补写一个 first-open 记录来消除该事实。

## 科学解释边界

全表读取不等于全表参与建模：随后 Domain A 左合并与 split 选择可以排除 B/ANCHOR 进入模型。此次发现本身不证明 B 被用于训练、调参或性能决策，也不重新判定 STEP5A 的 IID 指标。PATCH_01 仍是合法 Temperature_C 重复载入的修复，不被解释为第二次 test tuning。

但严格的“从未读取”封存要求与较窄的“从未参与开发或评价”要求不同。本次只能将前者判为不通过。不能把历史 ANCHOR target 的全表物化描述为从未读取；此次没有生成任何 ANCHOR 预测或指标。

## 按用户 STOP 规则执行

已创建要求的目录结构并保存审计证据。没有生成有效 preregistration、DOMAIN_B_FIRST_OPEN_RECORD、B 特征表、B 预测、OOD 指标、bootstrap CI 或图。没有修改 STEP5A 文件、重训模型、修改阈值或访问 Ni 数据；没有启动 STEP5C。

本次 Domain B composition/formal/converged/FAILED 数及性能均未计算，不用提示中的期望值代替实测。34 项最终评价问题中的性能、shift、距离、分层及守恒项均为 NOT EVALUATED。GO/NO-GO 为 NO-GO（协议关卡），不是模型性能不合格。

## 继续所需的协议决定

需明确批准并记录一个修订后的前瞻性评价协议：如保留历史全表读取偏离记录，将评价封存定义为“未用于开发、调参或 OOD 性能评价”，并先补充该较窄声明的审计。不能恢复原有“从未读取”事实；修改 loader 只能避免未来重复读取，不能修复历史。本次未擅自修改该定义。
'''
(O/'07_qa/STEP5B_DOMAIN_B_OOD_REPORT.md').write_text(report,encoding='utf-8')
(O/'07_qa/STEP5B_LEAKAGE_AND_SEAL_AUDIT.md').write_text(report,encoding='utf-8')
(O/'09_manifest/STAGE_A_AUDIT_EVIDENCE.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'hash_checks_passed':len(checks),'footer_rows':footer_rows,'status':summary['status'],'evaluation_executed':False}))
