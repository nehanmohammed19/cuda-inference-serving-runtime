"""Loopback HTTP load test: real server, worker, batching and CUDA execution."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import json
import threading
import time
import urllib.request
from runtime.backend import Backend
from runtime.model import Model,Config,toy_weights
from runtime.service import InferenceWorker,make_server
from benchmarks.run import summary


def experiment(mode):
    def factory():
        c=Config();model=Model(Backend('cuda'),c,toy_weights(c),pages=64,page_size=4,execution=mode)
        for batch in range(1,5):
            keys=[f'warm-{i}' for i in range(batch)]
            for key in keys:model.create(key)
            model.step([1]*batch,keys)
            for key in keys:model.release(key)
        return model
    worker=InferenceWorker(factory,queue_capacity=32,max_active=4,token_budget=4)
    server=make_server(worker,port=0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}/generate'
    def request(i):
        payload=json.dumps({'tokens':list(range([1,4,8,16][i%4])),'max_new_tokens':8}).encode()
        start=time.perf_counter()
        with urllib.request.urlopen(urllib.request.Request(url,data=payload),timeout=60) as response:
            result=json.load(response)
        result['http_latency_ms']=(time.perf_counter()-start)*1000
        return result
    try:
        start=time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(request,range(12)))
        elapsed=time.perf_counter()-start
        return {'mode':mode,'requests':12,'concurrency':4,'elapsed_s':elapsed,
                'output_tokens_per_s':sum(len(r['tokens']) for r in results)/elapsed,
                'http_latency_ms':summary([r['http_latency_ms'] for r in results]),
                'worker_stats':worker.snapshot(),'responses':results}
    finally:server.shutdown();server.server_close();thread.join();worker.close()


def main():
    records=[experiment(mode) for mode in ['eager','buffered','graph']]
    expected=[r['tokens'] for r in records[0]['responses']]
    assert all([r['tokens'] for r in record['responses']]==expected for record in records)
    path=Path('results/expansion/http-load.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'measured_at_utc':datetime.now(timezone.utc).isoformat(),
        'weights':'random toy seed 42','note':'one exploratory load run per mode; loopback HTTP',
        'runs':records},indent=2))
    print('Wrote',path)


if __name__=='__main__':main()
