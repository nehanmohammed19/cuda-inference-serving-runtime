"""Host Chrome traces and separately labeled CUDA-event intervals.

GPU event clocks are not calibrated to host wall time. Never draw invented GPU
timestamps on the host trace; export elapsed device durations as separate data.
"""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import time


class Profiler:
    def __init__(self,cp,enabled=False):
        self.cp,self.enabled=cp,enabled
        self.origin=time.perf_counter()
        self.events=[];self.device_events=[]
        self.nvtx=os.environ.get('BLOCKFORGE_NVTX')=='1'

    @contextmanager
    def region(self,name,**arguments):
        if not self.enabled:
            yield;return
        start=time.perf_counter()
        if self.nvtx:self.cp.cuda.nvtx.RangePush(name)
        try:yield
        finally:
            if self.nvtx:self.cp.cuda.nvtx.RangePop()
            end=time.perf_counter()
            self.events.append({'name':name,'cat':'host','ph':'X','pid':1,'tid':1,
                'ts':(start-self.origin)*1e6,'dur':(end-start)*1e6,'args':arguments})

    def begin_device(self,stream):
        if not self.enabled:return None
        event=self.cp.cuda.Event();event.record(stream);return event

    def end_device(self,name,start,stream):
        if start is None:return
        end=self.cp.cuda.Event();end.record(stream)
        self.device_events.append((name,start,end))

    def export(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        intervals=[]
        for name,start,end in self.device_events:
            end.synchronize()
            intervals.append({'name':name,'elapsed_ms':self.cp.cuda.get_elapsed_time(start,end)})
        path.write_text(json.dumps({'traceEvents':self.events,'displayTimeUnit':'ms',
            'metadata':{'clock':'host perf_counter; device durations separate',
                        'nvtx_enabled':self.nvtx},'cuda_event_intervals':intervals},indent=2))
        return intervals
