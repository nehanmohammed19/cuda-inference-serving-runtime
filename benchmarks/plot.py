"""Plot only recorded measurements. Re-run after replacing the JSON input."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('results'); data=json.loads((root/'measurements.json').read_text())
    plt.rcParams.update({'font.size':10,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    if 'kernels' in data:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for name in ['paged_cuda','paged_cuda_graph','cupy_dense_oracle']:
            rows=[r for r in data['kernels'] if r['implementation']==name]
            ax.plot([r['context'] for r in rows],[r['latency_ms']['median'] for r in rows],marker='o',label=name)
        ax.set(xlabel='Context length (tokens)',ylabel='CUDA-event latency (ms, median)',
               title='FP32 attention · batch 4 · 8 query / 2 KV heads · head dim 64')
        ax.set_xscale('log',base=2); ax.set_yscale('log'); ax.legend()
        fig.tight_layout(); fig.savefig(root/'attention.png'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,device in zip(axes,['cpu','cuda']):
        key=device+'_serving'
        if key not in data: ax.set_visible(False); continue
        rows=data[key]['trials']; policies=['serial','chunked','decode_first']
        values=[[r['output_tokens_per_s'] for r in rows if r['policy']==p] for p in policies]
        ax.bar(policies,[np.median(v) for v in values],color=['#577590','#43aa8b','#f9c74f'])
        ax.errorbar(range(3),[np.median(v) for v in values],
                    yerr=[[np.median(v)-min(v) for v in values],[max(v)-np.median(v) for v in values]],
                    fmt='none',color='#333333',capsize=4)
        ax.set(title=device.upper()+' · random tiny model',ylabel='Output tokens / second')
    fig.suptitle('Measured end-to-end workloads · bars: median; whiskers: trial range')
    fig.tight_layout(); fig.savefig(root/'serving.png'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    key='cuda_serving' if 'cuda_serving' in data else 'cpu_serving'
    for policy in ['serial','chunked','decode_first']:
        trial=next(r for r in data[key]['trials'] if r['policy']==policy)
        ax.step([t['elapsed_s'] for t in trial['trace']],
                [t['resident_pages'] for t in trial['trace']],where='post',label=policy)
    ax.set(xlabel='Measured wall time (s)',ylabel='Resident KV pages per layer',
           title='Physical cache residency · first recorded trial · '+key)
    ax.legend(); fig.tight_layout(); fig.savefig(root/'cache.png'); plt.close(fig)


if __name__=='__main__': main()
