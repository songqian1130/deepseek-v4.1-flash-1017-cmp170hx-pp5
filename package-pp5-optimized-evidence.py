"""Package raw acceptance, reviewed patches and reproducible entry points."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
base=Path(__file__).resolve().parent;root=base/'performance-hotspots-20261101'
summary=json.loads((root/'optimized-summary.json').read_text(encoding='utf-8'));assert summary['ok']
assert all(x['inter_token_tok_s']>=40 for x in summary['stable_baselines'])
assert all(x['passed'] for x in summary['quality_checks'])
entries={}
for file in root.rglob('*'):
    if file.is_file() and '__pycache__' not in file.parts and file.suffix not in ('.gz','.tgz','.zip','.pyc'):
        entries['evidence/'+file.relative_to(root).as_posix()]=file
sources=['activate-pp5-performance.py','remote-pp5-diag.py','profile-pp5-live.py','verify-pp5-performance-final.py','verify-pp5-no-degradation.py','verify-pp5-quality-battery.py','verify-pp5-launcher-contract.py','verify-pp5-reference-alignment.py','benchmark-pp5-mmap-chunks.py','summarize-pp5-optimized.py','package-pp5-optimized-evidence.py','start_vllm_v41_pp5_perf.sh','start_vllm_v41_pp5_full.sh','start_vllm_v41_pp5.sh','pp5-engram-mmap-run.sh','verify-pp5-ssd-async.py','benchmark-real-text.py','engram_mmap.py','pp5-engram-mmap-README.md']
for name in sources:entries['source/'+name]=base/name
entries['base-r6/disk_engram_mmap.py']=base/'performance-breakdown-20261031/r6-full-build/disk_engram_mmap.py'
entries['base-r4/engram.py']=base/'source-alignment-20261031/r4-runtime/runtime/vllm/models/deepseek_v4_1/common/engram.py'
entries['reference/launcher-source-contract.json']=base/'source-alignment-20261031/launcher-source-contract.json'
entries['reference/native-manifest.json']=base/'source-alignment-20261031/r4-native-manifest.json'
entries['README_ALIGNMENT.md']=base/'performance-breakdown-20261031/README_ALIGNMENT.md'
assert all(path.is_file() for path in entries.values())
manifest={'scope':'New optimization raw evidence and source; historical FULL GPU traces remain in Artifact Hub artifact1602. No model weights or native binaries.','entries':{name:{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in sorted(entries.items())}}
args.out.parent.mkdir(parents=True,exist_ok=True)
print(f'[HB] phase=pack-optimized files={len(entries)}',flush=True)
with zipfile.ZipFile(args.out,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as archive:
    for name,path in sorted(entries.items()):archive.write(path,name)
    archive.writestr('SHA256_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
with zipfile.ZipFile(args.out) as archive:
    assert archive.testzip() is None
    for name,row in manifest['entries'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==row['sha256']
sha=hashlib.sha256(args.out.read_bytes()).hexdigest()
receipt={'ok':True,'path':str(args.out.resolve()),'files':len(entries)+1,'bytes':args.out.stat().st_size,'sha256':sha}
args.out.with_suffix('.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt));print('PP5_OPTIMIZED_EVIDENCE_PACKAGE_DONE')
