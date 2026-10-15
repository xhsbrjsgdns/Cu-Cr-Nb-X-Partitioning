"""Stage E gate: re-verify the amended pre-open audit and the preregistration, then write
09_manifest/PREOPEN_GATE.json. Runs BEFORE any Domain-B target read."""
import json
from step5b_core import O, now, sha, load_json

def main():
    audit = load_json(O / '07_qa' / 'AMENDED_PREOPEN_AUDIT.json')
    assert audit['status'] == 'PASS_UNDER_AMENDMENT'
    assert audit['strict_never_read_status'] == 'FAIL_RETAINED'
    for e in audit['artifact_hashes']:
        if e.get('expected_sha256'):
            assert sha(e['path']) == e['expected_sha256'], e['path']
    pre_path = O / '01_preregistration' / 'STEP5B_PREREGISTRATION_V1.json'
    pre = load_json(pre_path)
    assert pre['protocol_amendment_approved'] is True
    for e in pre['frozen_artifacts'] + pre['prepared_inputs']:
        assert sha(e['path']) == e['sha256'], e['path']
    cp = O / '09_manifest' / 'CODE_SNAPSHOT_MANIFEST_STEP5B.json'
    assert sha(cp) == pre['code_manifest_sha256']
    for e in load_json(cp)['files']:
        assert sha(e['path']) == e['sha256'], e['path']
    assert not (O / '03_predictions' / 'DOMAIN_B_OOD_PREDICTIONS.parquet').exists()
    first_open_exists = (O / '01_preregistration' / 'DOMAIN_B_FIRST_OPEN_RECORD.json').exists()
    if first_open_exists:
        # PATCH_01 post-abort refresh: the opening physically happened once but produced
        # no prediction/metric artifact; the gate is re-issued with identical content plus
        # a refreshed preregistration hash. Requires the patch record.
        assert (O / '09_manifest' / 'STEP5B_PATCH_01.md').exists()
    gate = dict(status='PASS_UNDER_AMENDMENT', timestamp_utc=now(),
                preregistration_sha256=sha(pre_path),
                amended_audit_sha256=sha(O / '07_qa' / 'AMENDED_PREOPEN_AUDIT.json'),
                code_manifest_sha256=sha(cp),
                strict_never_read='FAIL_RETAINED (disclosed; see PROTOCOL_AMENDMENT_V1.md)',
                domain_b_artifacts_before_open=dict(predictions=0, metrics=0),
                first_open_record_exists=first_open_exists,
                gate_mode='post_abort_refresh' if first_open_exists else 'pre_open',
                anchor='sealed under strict and amended definitions',
                ni='sealed')
    (O / '09_manifest' / 'PREOPEN_GATE.json').write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding='utf-8')
    print('PREOPEN GATE: PASS_UNDER_AMENDMENT')


if __name__ == '__main__':
    main()
