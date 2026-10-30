from __future__ import annotations

import contextlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from decode_runtime import DecodeEventPool, CUDAGraphMLP, method_order, mlp_block
from popkv import LlamaSparseDecoder, RetrievalConfig, RetrievalCaches, DecodeDiagnostics, _DecodeStageTimer


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.post_attention_layernorm = torch.nn.LayerNorm(16)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.SiLU(), torch.nn.Linear(32, 16))


class MLPFairnessTests(unittest.TestCase):
    def test_common_mlp_preserves_values_and_input(self):
        torch.manual_seed(42)
        layer = Layer().eval()
        for x in [torch.randn(1, 1, 16), torch.randn(1, 1, 32)[..., ::2]]:
            original = x.clone()
            expected = x + layer.mlp(layer.post_attention_layernorm(x))
            torch.testing.assert_close(mlp_block(layer, x), expected, rtol=0, atol=0)
            torch.testing.assert_close(x, original, rtol=0, atol=0)

    def test_event_handles_created_and_initialized_before_tokens(self):
        created = []
        stream = SimpleNamespace(synchronize=lambda: None)
        class Event:
            def __init__(self, **kw):self.stamp=0;self.records=0;created.append(self)
            def record(self, passed_stream):
                self.records += 1
                self.stamp = Event.clock
                Event.clock += 1
            def synchronize(self):pass
            def elapsed_time(self, other):return other.stamp-self.stamp
        Event.clock=0
        with patch('torch.cuda.Event', Event), patch('torch.cuda.device', lambda _:contextlib.nullcontext()), patch('torch.cuda.current_stream', lambda _:stream):
            pool = DecodeEventPool(torch.device('cuda:1'), 6)
            self.assertEqual(len(created), 6)
            self.assertTrue(all(e.records==1 for e in created))
            for _ in range(2):
                timer=_DecodeStageTimer(torch.device('cuda:1'),pool)
                timer.mark('output_projection');timer.mark('mlp')
                diag=DecodeDiagnostics();timer.finish(diag)
                self.assertEqual(diag.mlp_ms,1)
                self.assertEqual(diag.gpu_decode_ms,2)
                self.assertEqual(diag.gpu_component_sum_ms,diag.gpu_decode_ms)
            self.assertEqual(len(created),6,'mark() or token2 allocated new CUDA events')

    def test_pool_overflow_fails_instead_of_allocating_timed_events(self):
        pool=DecodeEventPool.__new__(DecodeEventPool);pool.cursor=0;pool.events=[]
        with self.assertRaisesRegex(RuntimeError,'exhausted'):pool.record()

    def test_balanced_order_preserves_methods_and_positions(self):
        methods=('full','quest','fier','popkv-q','popkv-s')
        rows=[method_order(methods,i) for i in range(2*len(methods))]
        for row in rows:self.assertCountEqual(row,methods)
        for pos in range(len(methods)):
            for m in methods:self.assertEqual(sum(row[pos]==m for row in rows),2)
        with self.assertRaises(ValueError):method_order([],0)

    def test_paired_latency_analysis_uses_rounds_not_individual_tokens(self):
        from experiments.benchmark_mlp_fairness import paired_comparisons
        rows=[]
        for experiment in ['integrated','common_input_layer']:
            for rnd in range(4):
                for method,offset in [('full',0),('quest',2)]:
                    for value in [10+rnd,12+rnd]:
                        rows.append(dict(experiment=experiment,backend='eager',round=rnd,method=method,mlp_ms=value+offset))
        result=paired_comparisons(rows,['eager'],['full','quest'])
        self.assertEqual(len(result),2)
        for r in result:
            self.assertEqual(r['paired_rounds'],4)
            self.assertEqual(r['mean_difference_ms'],2)
            self.assertEqual(r['paired_bootstrap95_low_ms'],2)
            self.assertEqual(r['paired_bootstrap95_high_ms'],2)

    def test_graph_explicitly_requires_cuda(self):
        with self.assertRaisesRegex(ValueError,'CUDA device'):
            CUDAGraphMLP(Layer().eval(),torch.randn(1,1,16))

    def test_full_and_sparse_dense_layers_use_identical_mlp_and_logits(self):
        from transformers import LlamaConfig,LlamaForCausalLM
        torch.manual_seed(42)
        model=LlamaForCausalLM(LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128)).eval()
        decoder=LlamaSparseDecoder(model,RetrievalConfig(budget=4,full_layers=(0,1)))
        prefix=decoder.build_static_kv_cache(decoder.build_prefix_cache(torch.tensor([1,2,3,4])),reserve_tokens=2)
        calls=[];run=decoder._run_mlp
        def record(i,x):
            calls.append((i,tuple(x.shape),x.dtype,tuple(x.stride())))
            return run(i,x)
        decoder._run_mlp=record
        full,fd=decoder.decode(prefix,5,method='full')
        sparse,sd=decoder.decode(prefix,5,method='quest',retrieval_caches=RetrievalCaches('quest',{}))
        torch.testing.assert_close(full,sparse,atol=0,rtol=0)
        self.assertEqual(calls[:2],calls[2:]);self.assertEqual(len(calls),4)
        decoder._run_mlp=lambda i,x:x+decoder.layers[i].mlp(decoder.layers[i].post_attention_layernorm(x))
        reference,_=decoder.decode(prefix,5,method='full')
        torch.testing.assert_close(full,reference,atol=0,rtol=0)
        for d in [fd,sd]:self.assertAlmostEqual(d.gpu_component_sum_ms+d.gpu_unattributed_ms,d.gpu_decode_ms,places=7)
        with self.assertRaises(ValueError):decoder.set_mlp_backend('invalid')
        with self.assertRaisesRegex(ValueError,'CUDA device'):decoder.set_mlp_backend('cuda-graph')
        self.assertEqual(decoder.mlp_backend,'eager')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA device unavailable')
    def test_graph_replay_multiple_inputs_matches_eager(self):
        layer=Layer().eval().cuda();example=torch.randn(1,1,16,device='cuda')
        graph=CUDAGraphMLP(layer,example)
        for x in [example,torch.zeros_like(example),-example,2*example]:
            expected=mlp_block(layer,x)
            torch.testing.assert_close(graph(x),expected)
        with self.assertRaises(ValueError):graph(torch.randn(1,2,16,device='cuda'))

if __name__=='__main__':unittest.main()
