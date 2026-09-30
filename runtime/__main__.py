"""Run token-ID workloads or a local pretrained Llama checkpoint."""
import argparse
import json
from pathlib import Path
from .backend import Backend
from .model import Config,Model,toy_weights
from .checkpoint import load_checkpoint
from .engine import Engine,Request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--prompt',default='The capital of France is')
    parser.add_argument('--tokens',default='1,2,3',help='toy token IDs')
    parser.add_argument('--max-new-tokens',type=int,default=16)
    parser.add_argument('--requests',type=Path,help='JSON list: key,prompt (IDs),max_new_tokens,arrival_s')
    parser.add_argument('--policy',choices=['serial','chunked','decode_first'],default='decode_first')
    parser.add_argument('--pages',type=int,default=128)
    parser.add_argument('--page-size',type=int,default=16)
    parser.add_argument('--token-budget',type=int,default=32)
    parser.add_argument('--prefill-chunk',type=int,default=8)
    args=parser.parse_args(); tokenizer=None; raw={}
    if args.checkpoint:
        from tokenizers import Tokenizer
        c,w,raw=load_checkpoint(args.checkpoint)
        tokenizer=Tokenizer.from_file(str(args.checkpoint/'tokenizer.json'))
        prompt=tokenizer.encode(args.prompt).ids
    else:
        c=Config(); w=toy_weights(c); prompt=[int(t) for t in args.tokens.split(',')]
    model=Model(Backend(args.device),c,w,pages=args.pages,page_size=args.page_size)
    engine=Engine(model,policy=args.policy,token_budget=args.token_budget,prefill_chunk=args.prefill_chunk)
    inputs=json.loads(args.requests.read_text()) if args.requests else [
        {'key':'request-0','prompt':prompt,'max_new_tokens':args.max_new_tokens}]
    for item in inputs:
        eos=raw.get('eos_token_id')
        if isinstance(eos,list): raise ValueError('multiple EOS IDs are not supported by this CLI')
        engine.submit(Request(**({'eos_token_id':eos}|item)))
    result=[]
    for r in engine.run():
        result.append({'key':r.key,'status':r.status,'generated':r.generated,
            'text':tokenizer.decode(r.generated) if tokenizer else None,
            'ttft_ms':None if r.first_token_at is None else (r.first_token_at-r.submitted_at)*1000})
    print(json.dumps({'device':args.device,'weights':'pretrained-local' if tokenizer else 'random-toy',
                      'requests':result},indent=2))


if __name__=='__main__': main()
