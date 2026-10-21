"""Local token-ID serving with one GPU-owning worker and bounded admission.

HTTP threads never call CUDA. The worker owns the model, execution streams and
scheduler. A ticket carries output events back to each waiting HTTP connection.
"""
import argparse
from concurrent.futures import Future
from dataclasses import dataclass,field
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
from queue import Queue,Empty,Full
import threading
import time
import uuid
from .engine import Engine,Request


@dataclass
class Ticket:
    key:str
    prompt:list
    max_new_tokens:int
    result:Future=field(default_factory=Future)
    events:Queue=field(default_factory=Queue)
    cancelled:threading.Event=field(default_factory=threading.Event)
    published:int=0
    submitted_at:float=field(default_factory=time.perf_counter)


class InferenceWorker:
    def __init__(self,model_factory,queue_capacity=32,max_active=8,token_budget=16):
        self.factory=model_factory;self.inbox=Queue(queue_capacity)
        self.max_active=max_active;self.token_budget=token_budget
        self.stop=threading.Event();self.ready=Future();self.lock=threading.Lock()
        self.stats={'state':'starting','completed':0,'cancelled':0,'failed':0,'active':0,'queued':0}
        self.thread=threading.Thread(target=self._run,name='blockforge-device',daemon=True)
        self.thread.start();self.ready.result(timeout=120)

    def submit(self,prompt,max_new_tokens=16):
        if self.stop.is_set():raise RuntimeError('worker is stopped')
        ticket=Ticket(uuid.uuid4().hex,list(prompt),max_new_tokens)
        try:self.inbox.put_nowait(ticket)
        except Full:raise OverflowError('request queue is full') from None
        return ticket

    def snapshot(self):
        with self.lock:return dict(self.stats)

    def _finish_ticket(self,ticket,request):
        value={'id':ticket.key,'status':request.status,'tokens':list(request.generated),
            'ttft_ms':None if request.first_token_at is None else
                       (request.first_token_at-ticket.submitted_at)*1000,
            'elapsed_ms':(time.perf_counter()-ticket.submitted_at)*1000}
        ticket.result.set_result(value);ticket.events.put({'event':'done','data':value})
        with self.lock:self.stats['cancelled' if request.status=='cancelled' else 'completed']+=1

    def _run(self):
        tickets={};model=None
        try:
            model=self.factory();engine=Engine(model,max_active=self.max_active,
                                               token_budget=self.token_budget,policy='decode_first')
            with self.lock:self.stats['state']='ready'
            self.ready.set_result(True)
            while not self.stop.is_set():
                # Bound all outstanding work, including the scheduler wait queue.
                while len(tickets)<self.inbox.maxsize:
                    try:ticket=self.inbox.get_nowait()
                    except Empty:break
                    try:
                        request=Request(ticket.key,ticket.prompt,ticket.max_new_tokens,
                                        arrival_s=max(0,ticket.submitted_at-engine.start))
                        engine.submit(request);tickets[ticket.key]=ticket
                    except (ValueError,TypeError) as error:
                        ticket.result.set_exception(error)
                        ticket.events.put({'event':'error','data':str(error)})
                        with self.lock:self.stats['failed']+=1
                for key,ticket in tickets.items():
                    if ticket.cancelled.is_set():engine.cancel(key)
                if tickets:engine.tick()
                for key,ticket in list(tickets.items()):
                    request=engine.requests[key]
                    for token in request.generated[ticket.published:]:
                        ticket.events.put({'event':'token','data':int(token)})
                    ticket.published=len(request.generated)
                    if request.status in ['finished','cancelled']:
                        self._finish_ticket(ticket,request);del tickets[key];del engine.requests[key]
                # Profiling is opt-in; do not retain an unbounded service trace.
                if len(engine.trace)>1024:del engine.trace[:-1024]
                with self.lock:
                    self.stats.update(active=len(engine.active),queued=len(engine.waiting)+self.inbox.qsize())
                if not tickets:self.stop.wait(0.001)
        except BaseException as error:
            if not self.ready.done():self.ready.set_exception(error)
            with self.lock:self.stats.update(state='failed',error=str(error))
            self.stop.set()
        finally:
            # Resolve every outstanding future, including requests in the inbox.
            pending=list(tickets.values())
            while True:
                try:pending.append(self.inbox.get_nowait())
                except Empty:break
            for ticket in pending:
                if not ticket.result.done():ticket.result.set_exception(RuntimeError('worker stopped'))
                ticket.events.put({'event':'error','data':'worker stopped'})
            if model is not None:model.close()
            with self.lock:
                if self.stats['state']!='failed':self.stats['state']='stopped'

    def close(self):
        self.stop.set();self.thread.join(timeout=120)
        if self.thread.is_alive():raise TimeoutError('device worker did not stop')


def make_server(worker,host='127.0.0.1',port=8080):
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args):pass

        def respond(self,status,value):
            data=json.dumps(value).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)

        def do_GET(self):
            if self.path not in ['/health','/metrics']:
                self.respond(404,{'error':'unknown endpoint'});return
            stats=worker.snapshot()
            self.respond(200 if stats['state']=='ready' else 503,stats)

        def do_POST(self):
            if self.path!='/generate':self.respond(404,{'error':'unknown endpoint'});return
            ticket=None
            try:
                length=int(self.headers.get('Content-Length','0'))
                if length<1 or length>65536:raise ValueError('body must be 1..65536 bytes')
                payload=json.loads(self.rfile.read(length))
                prompt=payload['tokens'];count=payload.get('max_new_tokens',16)
                if not isinstance(prompt,list) or not prompt or any(type(t)!=int for t in prompt):
                    raise ValueError('tokens must be a nonempty integer list')
                if type(count)!=int or not 0<=count<=256:raise ValueError('max_new_tokens must be 0..256')
                ticket=worker.submit(prompt,count)
                if payload.get('stream',False):
                    self.send_response(200);self.send_header('Content-Type','text/event-stream')
                    self.send_header('Connection','close');self.end_headers();self.close_connection=True
                    while True:
                        event=ticket.events.get(timeout=120)
                        self.wfile.write(('event: '+event['event']+'\ndata: '+json.dumps(event['data'])+'\n\n').encode())
                        self.wfile.flush()
                        if event['event'] in ['done','error']:break
                else:self.respond(200,ticket.result.result(timeout=120))
            except OverflowError as error:self.respond(429,{'error':str(error)})
            except (ValueError,TypeError,KeyError) as error:self.respond(400,{'error':str(error)})
            except (BrokenPipeError,ConnectionResetError):
                if ticket is not None:ticket.cancelled.set()
            except (TimeoutError,Empty):
                if ticket is not None:ticket.cancelled.set()
                if not self.close_connection:self.respond(504,{'error':'request timed out'})
            except RuntimeError as error:self.respond(503,{'error':str(error)})
    return ThreadingHTTPServer((host,port),Handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--execution',choices=['eager','buffered','graph'],default='eager')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--port',type=int,default=8080)
    args=parser.parse_args()
    def factory():
        from .backend import Backend
        from .model import Model,Config,toy_weights
        if args.checkpoint:
            from .checkpoint import load_checkpoint
            config,weights,_=load_checkpoint(args.checkpoint)
        else:config=Config();weights=toy_weights(config)
        return Model(Backend(args.device),config,weights,pages=128,execution=args.execution)
    worker=InferenceWorker(factory);server=make_server(worker,port=args.port)
    print(f'BlockForge listening on http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();worker.close()


if __name__=='__main__':main()
