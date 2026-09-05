"""Audit the shipped measurements and the accounting used by the public tables."""
import csv
import json
import unittest
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from experiments.summarize_five_methods import (
    GENERATION_STAGES, additive_table, generation_breakdown, score_time,
)

ROOT = Path(__file__).resolve().parents[1]
PG = ROOT / 'results/five_methods_split_attention_32k_b4096_t256_blocks10'
LB = ROOT / 'results/longbench_950_b4096'
TASK_COUNTS = {'qasper': 200, 'multifieldqa_en': 150, 'hotpotqa': 200, 'gov_report': 200, 'triviaqa': 200}
METHODS = {'full', 'quest', 'fier-author', 'popkv-q', 'popkv-s'}


def csv_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


class PublishedResultsTests(unittest.TestCase):
    def test_longbench_completeness_and_recomputed_means(self):
        with (LB / 'raw_results.jsonl').open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(len(rows), 4750)
        self.assertEqual({r['status'] for r in rows}, {'ok'})
        self.assertEqual({r['method'] for r in rows}, METHODS)
        groups = defaultdict(list)
        samples = defaultdict(dict)
        keys = set()
        for r in rows:
            key = (r['method'], r['dataset'], r['sample_id'])
            self.assertNotIn(key, keys)
            keys.add(key)
            groups[(r['method'], r['dataset'])].append(r['score'])
            samples[r['method']][(r['dataset'], r['sample_id'])] = r['input_sha256']
        for m in METHODS:
            self.assertEqual(dict(Counter(r['dataset'] for r in rows if r['method'] == m)), TASK_COUNTS)
            self.assertEqual(samples[m], samples['full'])
        per_task = {(m,t):100*sum(v)/len(v) for (m,t),v in groups.items()}
        for r in csv_rows(LB / 'scores_summary.csv'):
            m=r['method']
            self.assertEqual(int(r['total_success']), 950)
            self.assertAlmostEqual(float(r['task_average']),sum(per_task[m,t] for t in TASK_COUNTS)/5, places=10)
            self.assertAlmostEqual(float(r['mean_retention_percent']),sum(per_task[m,t]/per_task['full',t]*100 for t in TASK_COUNTS)/5,places=10)
        manifest=json.loads((LB/'sample_manifest.json').read_text())
        self.assertEqual({(r['dataset'],r['sample_id']):r['input_sha256'] for r in manifest},samples['full'])
        config=json.loads((LB/'experiment_config.json').read_text())
        self.assertEqual(set(config['resolved_datasets']),set(TASK_COUNTS))

    def test_pg19_raw_nll_and_exhaustive_time_accounting(self):
        import math
        rows=csv_rows(PG/'raw/samples.csv')
        self.assertEqual(len(rows),12800)
        for s in csv_rows(PG/'raw/summary.csv'):
            selected=[r for r in rows if r['method']==s['method']]
            self.assertEqual(len(selected),2560)
            self.assertEqual(dict(Counter(int(r['block']) for r in selected)),dict.fromkeys(range(10),256))
            self.assertAlmostEqual(math.exp(sum(float(r['nll']) for r in selected)/2560),float(s['perplexity']),places=10)
            self.assertAlmostEqual(sum(generation_breakdown(s).values()),float(s['generation_step_ms_per_token']),places=8)
            broken=dict(s)
            broken['generation_step_ms_per_token']=str(float(s['generation_step_ms_per_token'])+1)
            with self.assertRaises(ValueError):
                generation_breakdown(broken)

    def test_legacy_and_split_score_accounting(self):
        self.assertEqual(score_time({'score_ms_per_token':'3'}),3)
        self.assertEqual(score_time({'score_ms_per_token':'2','retrieval_query_encode_ms_per_token':'1','retrieval_score_ms_per_token':'2'}),3)
        with self.assertRaises(KeyError):
            score_time({'retrieval_score_ms_per_token':'2'})

    def test_printed_components_sum_exactly_to_printed_total(self):
        rows=csv_rows(PG/'generation_breakdown.csv')
        rows=[{k:v if k=='method' else float(v) for k,v in r.items()} for r in rows]
        table=additive_table(rows,GENERATION_STAGES,'total','Total token generation')
        values=[line.split('|')[2:-1] for line in table.splitlines()[2:]]
        for column in range(5):
            parts=[Decimal(v[column].strip().strip('*')) for v in values]
            self.assertEqual(sum(parts[:-1]),parts[-1])
        readme=(ROOT/'README.md').read_text()
        self.assertIn(table,readme)
        self.assertIn(table,(PG/'report.md').read_text())

    def test_only_published_longbench_tasks_are_executable(self):
        from experiments.evaluate_longbench import PROMPTS, MAX_GEN, score_prediction
        self.assertEqual(set(PROMPTS),set(TASK_COUNTS))
        self.assertEqual(set(MAX_GEN),set(TASK_COUNTS))
        with self.assertRaises(ValueError):
            score_prediction('unsupported_task','answer',['answer'])


if __name__ == '__main__':
    unittest.main()
