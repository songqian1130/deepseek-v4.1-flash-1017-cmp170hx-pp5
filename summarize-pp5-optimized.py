"""Derive optimized-service acceptance from preserved raw measurements."""
import argparse
from collections import Counter
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--profile',default='r9');args=p.parse_args();root=args.directory;profile=args.profile
def read(path):return json.loads(path.read_text(encoding='utf-8'))
ab=[]
for name in ['control-baseline-1','lazy-baseline-1','lazy-baseline-2','control-baseline-2','lazy-final']:
    d=read(root/'r7-lazy-receive/results'/f'{name}.json')
    assert d['usage']['completion_tokens']==256 and d['usage']['prompt_tokens']==55
    ab.append({'label':name,'decode_tok_s':d['decode_tok_s'],'usage':d['usage']})
texts=[read(root/'r7-lazy-receive/results'/f"{x['label']}.json")['text'] for x in ab]
assert len(set(texts))==1
long_texts=[read(root/f'{candidate}-live/{label}.json')['text'] for candidate in ['r8',profile] for label in ['baseline-1','sampled','baseline-2']]
assert len(set(long_texts))==1
live=read(root/f'{profile}-live-stable/summary.json');rates=[];io=[]
for req in live['requests']:
    raw=read(root/f'{profile}-live-stable'/f"{req['label']}.json")
    assert raw['completion_tokens']==1536 and raw['usage']['prompt_tokens']==55
    if req['label'].startswith('baseline-'):
        rates.append({k:raw[k] for k in ['label','decode_tok_s','inter_token_tok_s','end_to_end_tok_s','ttft_s','completion_tokens']})
    bounds=live['boundaries'][req['label']];workers=[]
    for rank,before in bounds['before']['workers'].items():
        after=bounds['after']['workers'][rank]
        seconds=bounds['after']['at']-bounds['before']['at']
        workers.append({'rank':int(rank),'read_bytes':after['io']['read_bytes']-before['io']['read_bytes'],'major_faults':after['major_faults']-before['major_faults'],'cpu_ms_per_s':1000*((after['user_ticks']+after['sys_ticks'])-(before['user_ticks']+before['sys_ticks']))/live['clock_ticks_per_second']/seconds})
    io.append({'label':req['label'],'workers':workers})
assert len(rates)==2 and all(x['inter_token_tok_s']>=40 for x in rates),rates
parity=[]
for path in [root/'r7-lazy-receive/quality-parity/summary.json',root/'r8-production/quality-parity/summary.json',root/f'{profile}-production/quality-parity/summary.json']:
    d=read(path);assert d['ok'] and all(x['passed'] for x in d['pairs'])
    parity.append({'path':str(path.relative_to(root)),'cases':len(d['pairs']),'recorded_entries':sum(x['logprob_entries'] for x in d['pairs']),'max_logprob_abs_diff':max(x['max_logprob_abs_diff'] for x in d['pairs'])})
quality=read(root/f'{profile}-quality/summary.json');assert quality['ok']
final=read(root/f'{profile}-production/final-acceptance.json');assert final['ok']
source=read(root/f'{profile}-production/source-verification.json');assert source['ok']
first_runs=[]
for candidate in ['r8',profile]:
    data=read(root/f'{candidate}-live/summary.json')
    first=data['requests'][0];bounds=data['boundaries']['baseline-1'];workers=[]
    for rank,before in bounds['before']['workers'].items():
        after=bounds['after']['workers'][rank]
        workers.append({'rank':int(rank),'read_bytes':after['io']['read_bytes']-before['io']['read_bytes'],'major_faults':after['major_faults']-before['major_faults']})
    first_runs.append({'profile':candidate,'request':first,'io':workers})
stacks=[]
for path in sorted((root/f'{profile}-live').glob('rank*.raw')):
    leaves=Counter();total=0;malformed=0
    for line in path.read_text(encoding='utf-8',errors='replace').splitlines():
        try:
            stack,count=line.rsplit(' ',1);count=int(count)
        except ValueError:malformed+=1;continue
        # Keep exact thread/leaf evidence. Idle library threads are separate from
        # the model main thread and must not be treated as model critical time.
        frames=stack.split(';');leaves[(frames[0],frames[-1])]+=count;total+=count
    stacks.append({'rank':int(path.stem.replace('rank','')),'samples_all_threads':total,'malformed_lines':malformed,'top_thread_leaf':[{'thread':t,'leaf':leaf,'samples':n} for (t,leaf),n in leaves.most_common(15)]})
result={'ok':True,'profile':profile,'r7_same_instance_ab':ab,'r7_all_five_texts_equal':True,'long_output_parity':{'requests':len(long_texts),'tokens_per_request':1536,'all_texts_equal':True},'stable_baselines':rates,'mean_decode_tok_s':sum(x['decode_tok_s'] for x in rates)/2,'mean_inter_token_tok_s':sum(x['inter_token_tok_s'] for x in rates)/2,'io_and_cpu':io,'first_runs':first_runs,'parity':parity,'quality_checks':quality['checks'],'stack_summary':stacks,'live_image_id':final['image_id'],'scope':['Single request decode; includes reasoning tokens','Stable speed uses independent unsampled requests; first-run page faults and sampled requests remain separate','Five-prompt score parity and eight complete-answer checks are finite acceptance, not universal intelligence proof','No 1M end-to-end, vision/tool or broad cold-cache throughput acceptance']}
(root/'optimized-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ['ok','mean_decode_tok_s','mean_inter_token_tok_s','parity','live_image_id']},indent=2));print('PP5_OPTIMIZED_SUMMARY_DONE')
