"""Continuous batching with bounded prefill and explicit page admission.

One tick emits at most one decode token per request. Prefills advance in bounded
microsteps, sharing dense GEMMs with other ready rows. This baseline recomputes
metadata on the CPU and synchronizes each microstep; there is no overlap claim.
"""
from collections import deque
from dataclasses import dataclass, field
import math
import time


@dataclass
class Request:
    key: str
    prompt: list
    max_new_tokens: int = 16
    eos_token_id: int | None = None
    arrival_s: float = 0.
    generated: list = field(default_factory=list)
    cursor: int = 0
    status: str = 'waiting'
    submitted_at: float = 0.
    first_token_at: float | None = None
    token_times: list = field(default_factory=list)
    finished_at: float | None = None


class Engine:
    def __init__(self, model, max_active=8, token_budget=32, prefill_chunk=8, policy='chunked'):
        if min(max_active,token_budget,prefill_chunk)<=0: raise ValueError('invalid scheduler limits')
        if policy not in ['serial','chunked','decode_first']: raise ValueError('invalid policy')
        self.model=model; self.max_active=1 if policy=='serial' else max_active
        self.token_budget=token_budget; self.prefill_chunk=prefill_chunk; self.policy=policy
        self.waiting=deque(); self.active=[]; self.requests={}; self.reserved={}
        self.trace=[]; self.start=time.perf_counter(); self.rotation=0

    def submit(self, request):
        r=request; c=self.model.config; cache=self.model.caches[0]
        if r.key in self.requests: raise ValueError('duplicate request ID')
        if not r.prompt or r.max_new_tokens<0 or r.arrival_s<0: raise ValueError('invalid request')
        if r.status!='waiting' or r.cursor or r.generated: raise ValueError('request already used')
        if any(t<0 or t>=c.vocab_size for t in r.prompt): raise ValueError('invalid prompt token')
        total=len(r.prompt)+max(0,r.max_new_tokens-1)
        if total>c.max_position_embeddings: raise ValueError('request exceeds context limit')
        pages=math.ceil(total/cache.page_size)
        if pages>len(cache.refs): raise ValueError('request cannot fit in page pool')
        r.submitted_at=self.start+r.arrival_s
        self.requests[r.key]=r
        self.waiting.append(r)

    def _finish(self,r,status='finished'):
        # release waits for device completion before a physical page can be reused.
        if r.key in self.reserved:
            self.model.release(r.key); del self.reserved[r.key]
            self.active.remove(r)
        r.status=status; r.finished_at=time.perf_counter()

    def cancel(self,key):
        r=self.requests[key]
        if r.status in ['finished','cancelled']: return
        if r in self.waiting: self.waiting.remove(r)
        self._finish(r,'cancelled')

    def _admit(self):
        cache=self.model.caches[0]; available=len(cache.refs)-sum(self.reserved.values())
        # Scan the queue so a large request does not block every smaller request.
        for r in list(self.waiting):
            if len(self.active)>=self.max_active: break
            if time.perf_counter()<r.submitted_at: continue
            pages=math.ceil((len(r.prompt)+max(0,r.max_new_tokens-1))/cache.page_size)
            if pages>available: continue
            self.waiting.remove(r)
            if r.max_new_tokens==0: self._finish(r); continue
            self.model.create(r.key); self.reserved[r.key]=pages
            available-=pages; r.status='prefill'; self.active.append(r)

    def tick(self):
        self._admit()
        if not self.active: return bool(self.waiting)
        remaining=self.token_budget; prefill_used={}; decoded=set(); now=time.perf_counter()
        # Rotating priority avoids starving later requests when budget < batch size.
        order=self.active[self.rotation:]+self.active[:self.rotation]
        self.rotation=(self.rotation+1)%max(1,len(self.active))
        if self.policy=='decode_first': order.sort(key=lambda r:r.status!='decode')
        used=0
        while remaining:
            rows=[]
            for r in order:
                if r not in self.active: continue
                if r.status=='decode':
                    if r.key in decoded: continue
                elif prefill_used.get(r.key,0)>=self.prefill_chunk: continue
                rows.append(r)
                if len(rows)==remaining: break
            if not rows: break
            tokens=[r.prompt[r.cursor] if r.status=='prefill' else r.generated[-1] for r in rows]
            logits=self.model.step(tokens,[r.key for r in rows])
            choices=self.model.backend.numpy(self.model.xp.argmax(logits,axis=-1)).tolist()
            self.model.backend.sync()
            completed=time.perf_counter()
            for r,token in zip(rows,choices):
                if r.status=='prefill':
                    r.cursor+=1; prefill_used[r.key]=prefill_used.get(r.key,0)+1
                    if r.cursor<len(r.prompt): continue
                    r.status='decode'
                decoded.add(r.key)
                r.generated.append(int(token)); r.token_times.append(completed)
                if r.first_token_at is None: r.first_token_at=completed
                if len(r.generated)>=r.max_new_tokens or token==r.eos_token_id: self._finish(r)
            remaining-=len(rows); used+=len(rows)
        self.trace.append({'elapsed_s':time.perf_counter()-self.start,'step_s':time.perf_counter()-now,
                           'processed_tokens':used,'active':len(self.active),
                           'resident_pages':len(self.model.caches[0].refs)-len(self.model.caches[0].free)})
        return bool(self.active or self.waiting)

    def run(self):
        while self.tick():
            if not self.active and self.waiting:
                delay=min(r.submitted_at for r in self.waiting)-time.perf_counter()
                if delay>0: time.sleep(min(delay,0.01))
        return list(self.requests.values())
