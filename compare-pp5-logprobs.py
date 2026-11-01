"""Capture or compare deterministic PP5 completions and top-5 token scores."""
import argparse
import json
from pathlib import Path
import urllib.request

PROMPTS = [
    '算出17乘以23，并写出计算过程。',
    '所有A都是B，所有B都是C。能否推出所有A都是C？说明理由。',
    'Write a Python function that returns the median of a non-empty list without changing the original list.',
    'A bag has 3 red balls and 2 blue balls. Draw two without replacement. What is the probability both are red? Show the fraction.',
    '阅读后回答：甲先于乙到达，丙晚于乙到达，丁早于甲到达。请给出四人的到达顺序。',
]


def post(base, body):
    req = urllib.request.Request(
        base.rstrip('/') + '/v1/chat/completions',
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.loads(response.read())


def flatten(value):
    rows = []
    if isinstance(value, list):
        for item in value:
            rows.extend(flatten(item))
    elif isinstance(value, dict):
        if isinstance(value.get('logprob'), (int, float)):
            rows.append((value.get('token'), float(value['logprob'])))
        for key, item in value.items():
            if key != 'logprob' and isinstance(item, (list, dict)):
                rows.extend(flatten(item))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for index, prompt in enumerate(PROMPTS):
        body = {
            'model': 'deepseek-v4.1-flash-1017',
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0,
            'max_tokens': 64,
            'ignore_eos': True,
            'logprobs': True,
            'top_logprobs': 5,
            'stream': False,
        }
        actual = post(args.base, body)
        (args.out / f'case{index}.json').write_text(json.dumps(actual, ensure_ascii=False, indent=2), encoding='utf-8')
        if args.reference_dir is None:
            continue
        reference = json.loads((args.reference_dir / f'case{index}.json').read_text(encoding='utf-8'))
        a, b = reference['choices'][0], actual['choices'][0]
        la, lb = flatten(a.get('logprobs')), flatten(b.get('logprobs'))
        assert la and len(la) == len(lb), (index, len(la), len(lb))
        tokens_equal = all(x[0] == y[0] for x, y in zip(la, lb))
        max_diff = max(abs(x[1] - y[1]) for x, y in zip(la, lb))
        item = {
            'case': index,
            'message_equal': a.get('message') == b.get('message'),
            'logprob_entries': len(la),
            'top_tokens_equal': tokens_equal,
            'max_logprob_abs_diff': max_diff,
            'passed': a.get('message') == b.get('message') and tokens_equal and max_diff <= 1e-5,
        }
        results.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    if args.reference_dir is not None:
        summary = {'ok': all(item['passed'] for item in results), 'pairs': results}
        (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        if not summary['ok']:
            raise SystemExit('PP5 top-5 logprob parity failed')
        print('PP5_TOP5_LOGPROB_PARITY_DONE', flush=True)
    else:
        print('PP5_TOP5_LOGPROB_REFERENCE_DONE', flush=True)


if __name__ == '__main__':
    main()
