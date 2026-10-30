"""Matched MLP comparison: integrated decode plus a common-input layer control.

This fixed-context diagnostic does not alter archived PG19 latency or PPL.
"""
from __future__ import annotations

import argparse,csv,hashlib,json,platform,statistics,sys,random
from datetime import datetime,timezone
from pathlib import Path


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='meta-llama/Llama-3.1-8B')
    p.add_argument('--methods',default='full,quest,fier,popkv-q,popkv-s')
    p.add_argument('--mlp-backends',default='eager,cuda-graph')
    p.add_argument('--context-length',type=int,default=32768)
    p.add_argument('--budget',type=int,default=4096)
    p.add_argument('--rounds',type=int,default=10)
    p.add_argument('--steps',type=int,default=16)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--control-repeats',type=int,default=20)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--text-file',default='data/pg19/test/6941.txt')
    p.add_argument('--token-ids',help='Optional saved1D CPU torch.long tensor; overrides text file')
    p.add_argument('--dense-backend',choices=['grouped','flash'],default='grouped')
    p.add_argument('--dtype',choices=['bfloat16','float16','float32'],default='bfloat16')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--output-dir')
    return p.parse_args(argv)


def save(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def paired_comparisons(rows, backends, methods, seed=42):
    """Pair method and Full within round; bootstrap rounds, never token samples."""
    if 'full' not in methods:
        return []
    output=[]
    for experiment in ['integrated','common_input_layer']:
        for backend in backends:
            grouped={}
            for r in rows:
                if r['experiment']==experiment and r['backend']==backend:
                    grouped.setdefault((r['round'],r['method']),[]).append(r['mlp_ms'])
            rounds=sorted({rnd for rnd,m in grouped})
            for method in methods:
                if method=='full':continue
                pairs=[(statistics.mean(grouped[rnd,method]),statistics.mean(grouped[rnd,'full'])) for rnd in rounds]
                delta=[a-b for a,b in pairs];baseline=statistics.mean(b for a,b in pairs)
                rng=random.Random(seed)
                boot=sorted(statistics.mean(rng.choices(delta,k=len(delta))) for _ in range(5000))
                output.append(dict(experiment=experiment,backend=backend,method=method,reference='full',paired_rounds=len(pairs),
                    mean_difference_ms=statistics.mean(delta),relative_difference_percent=100*statistics.mean(delta)/baseline,
                    paired_bootstrap95_low_ms=boot[124],paired_bootstrap95_high_ms=boot[4874],bootstrap_unit='round',bootstrap_seed=seed))
    return output


def main(argv=None):
    args=arguments(argv)
    if min(args.rounds,args.steps,args.warmup,args.control_repeats)<1:raise ValueError('repeat counts must be positive')
    if args.context_length<2 or args.budget<2:raise ValueError('context and budget must be at least2')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=Path(args.output_dir or f'results/mlp_fairness_{stamp}')
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'use a fresh output directory: {out}')
    out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    pre=dict(status='Checking',python=sys.version,platform=platform.platform(),arguments=vars(args),
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'popkv.py',root/'decode_runtime.py',Path(__file__)]},
        device_nodes=[str(p) for p in Path('/dev').glob('nvidia*')])
    try:
        import torch
        pre.update(torch=torch.__version__,cuda_build=torch.version.cuda,cuda_available=torch.cuda.is_available())
    except ImportError as e:
        pre.update(status='Unavailable',reason=str(e));save(out/'preflight.json',pre);print(json.dumps(pre,indent=2));return 2
    if not torch.cuda.is_available():
        pre.update(status='Unavailable',reason='No accessible NVIDIA CUDA device; no GPU latency measured')
        save(out/'preflight.json',pre);print(json.dumps(pre,indent=2));return 2
    pre.update(status='Ready',gpu=torch.cuda.get_device_name(0))
    save(out/'preflight.json',pre)
    if args.check_only:print(json.dumps(pre,indent=2));return 0
    from popkv import LlamaSparseDecoder,RetrievalConfig,load_hf_model,METHODS
    from decode_runtime import method_order,DecodeEventPool,mlp_block
    methods=tuple(args.methods.split(','));backends=tuple(args.mlp_backends.split(','))
    if not methods or len(set(methods))!=len(methods) or set(methods)-set(METHODS):raise ValueError('invalid or duplicate methods')
    if not backends or len(set(backends))!=len(backends) or set(backends)-{'eager','cuda-graph'}:raise ValueError('invalid or duplicate MLP backends')
    backends=tuple(b for b in ('eager','cuda-graph') if b in backends)
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    model,tokenizer=load_hf_model(args.model,dtype=args.dtype,device_map='cuda:0',attention='sdpa')
    decoder=LlamaSparseDecoder(model,RetrievalConfig(budget=args.budget))
    decoder.experimental_full_sdpa=args.dense_backend=='flash'
    if args.token_ids:
        ids=torch.load(args.token_ids,map_location='cpu',weights_only=True)
        if not isinstance(ids,torch.Tensor) or ids.ndim!=1 or ids.dtype!=torch.long:raise ValueError('token-ids must be a1D torch.long tensor')
    else:
        ids=torch.tensor(tokenizer(Path(args.text_file).read_text(),add_special_tokens=False)['input_ids'],dtype=torch.long)
    if len(ids)<args.context_length:raise ValueError('input is shorter than requested context')
    ids=ids[:args.context_length].clone();torch.save(ids,out/'input_ids.pt')
    save(out/'input_contract.json',dict(token_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        context_length=len(ids),batch=1,query_tokens=1,mlp_scope='post-attention norm + MLP + residual; graph includes input copy',
        model_commit=getattr(model.config,'_commit_hash',None),fixed_context=True,append=False,
        actual_backend_order=list(backends),logit_check_rtol=0.02,logit_check_atol=0.02,
        notes='Same prefix and one teacher-forced token for every method/round; fixed-context diagnostic, not PG19 PPL reproduction'))
    prefix=decoder.build_prefix_cache(ids[:-1],reserve_tokens=2,chunk_size=1024)
    rows=[];contracts=[];reference_logits={};common_input=None;graph_comparisons=0
    target_layer=min(2,len(decoder.layers)-1)
    pool=DecodeEventPool(decoder.device,4)
    def elapsed_call(fn,repeats=1):
        pool.reset();a=pool.record()
        for _ in range(repeats):result=fn()
        b=pool.record();b.synchronize()
        return result,float(a.elapsed_time(b))/repeats
    with (out/'raw.csv').open('w',newline='') as raw:
        fields=['experiment','backend','round','position','method','sample','mlp_ms','decode_ms']
        writer=csv.DictWriter(raw,fieldnames=fields);writer.writeheader()
        def record(row):rows.append(row);writer.writerow(row);raw.flush()
        try:
            with torch.inference_mode():
                for backend in backends:
                    decoder.set_mlp_backend(backend)  # Capture outside all measured intervals.
                    for rnd in range(args.rounds):
                        order=method_order(methods,rnd)
                        for position,method in enumerate(order):
                            kv=decoder.build_static_kv_cache(prefix,reserve_tokens=2)
                            retrieval=None if method=='full' else decoder.build_retrieval_caches(kv,method=method,reserve_tokens=2)
                            original=decoder._run_mlp
                            captured={}
                            def inspect(i,x):
                                captured[i]=dict(layer=i,shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),device=str(x.device))
                                nonlocal common_input
                                if i==target_layer and common_input is None:common_input=x.detach().clone()
                                return original(i,x)
                            decoder._run_mlp=inspect
                            for _ in range(args.warmup):logits,_=decoder.decode(kv,ids[-1],method=method,retrieval_caches=retrieval)
                            decoder._run_mlp=original
                            contracts.append(dict(backend=backend,round=rnd,method=method,layers=list(captured.values())))
                            if backend=='eager':reference_logits[method]=logits.detach().cpu()
                            else:
                                if method not in reference_logits:
                                    decoder._run_mlp=lambda i,x:mlp_block(decoder.layers[i],x)
                                    eager_logits,_=decoder.decode(kv,ids[-1],method=method,retrieval_caches=retrieval)
                                    decoder._run_mlp=original
                                    reference_logits[method]=eager_logits.detach().cpu()
                                torch.testing.assert_close(logits.cpu(),reference_logits[method],atol=0.02,rtol=0.02)
                                graph_comparisons+=1
                            for step in range(args.steps):
                                (_,diag),decode_ms=elapsed_call(lambda:decoder.decode(kv,ids[-1],method=method,retrieval_caches=retrieval))
                                record(dict(experiment='integrated',backend=backend,round=rnd,position=position,method=method,sample=step,mlp_ms=diag.mlp_ms,decode_ms=decode_ms))
                            if args.profile and rnd==0:
                                def profiled(i,x):
                                    with torch.profiler.record_function(f'popkv::mlp.layer{i}'):
                                        return original(i,x)
                                decoder._run_mlp=profiled
                                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
                                    decoder.decode(kv,ids[-1],method=method,retrieval_caches=retrieval)
                                decoder._run_mlp=original
                                prof.export_chrome_trace(str(out/f'profile_{backend}_{method}.json'))
                            del retrieval,kv
                        # Common-input MLP control: same tensor, weights and callable across method labels.
                        expected=mlp_block(decoder.layers[target_layer],common_input)
                        for position,method in enumerate(order):
                            fn=lambda:decoder._run_mlp(target_layer,common_input)
                            for _ in range(args.warmup):actual=fn()
                            torch.testing.assert_close(actual,expected,atol=0.02,rtol=0.02)
                            _,ms=elapsed_call(fn,args.control_repeats)
                            record(dict(experiment='common_input_layer',backend=backend,round=rnd,position=position,method=method,sample=0,mlp_ms=ms,decode_ms='N/A'))
        except Exception as exc:
            save(out/'failure.json',dict(status='Failed',type=type(exc).__name__,reason=str(exc)))
            raise
    save(out/'mlp_input_contracts.json',contracts)
    first=contracts[0]['layers']
    assert all(c['layers']==first for c in contracts),'MLP shape/dtype/stride/device differs by method'
    summary=[]
    for experiment in ['integrated','common_input_layer']:
        for backend in backends:
            for method in methods:
                vals=[r['mlp_ms'] for r in rows if r['experiment']==experiment and r['backend']==backend and r['method']==method]
                summary.append(dict(experiment=experiment,backend=backend,method=method,samples=len(vals),mean_ms=statistics.mean(vals),median_ms=statistics.median(vals),stdev_ms=statistics.stdev(vals) if len(vals)>1 else 0.0))
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    paired=paired_comparisons(rows,backends,methods,args.seed)
    if paired:
        with (out/'paired_comparison.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    save(out/'validation.json',dict(status='Passed',same_mlp_input_contract=True,graph_vs_eager_values_checked=graph_comparisons>0,graph_logit_checks=graph_comparisons,
        force_equal_latency=False,latency_equality='Inspect paired rounds and common-input control; equality is not an acceptance requirement',
        samples=len(rows),rounds=args.rounds))
    print(out)
    return 0

if __name__=='__main__':raise SystemExit(main())
