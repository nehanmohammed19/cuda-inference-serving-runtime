import json
import threading
import unittest
import urllib.request
import urllib.error
from runtime.backend import Backend
from runtime.model import Model,Config,toy_weights
from runtime.service import InferenceWorker,make_server


class ServiceTests(unittest.TestCase):
    def setUp(self):
        def factory():
            c=Config();return Model(Backend(),c,toy_weights(c),pages=32,page_size=4)
        self.worker=InferenceWorker(factory,queue_capacity=8,token_budget=2)

    def tearDown(self):self.worker.close()

    def test_concurrent_submission_and_cancellation(self):
        tickets=[self.worker.submit([1,2,3],4) for _ in range(4)]
        outputs=[t.result.result(timeout=10)['tokens'] for t in tickets]
        self.assertTrue(all(o==outputs[0] for o in outputs))
        ticket=self.worker.submit([1]*40,40);ticket.cancelled.set()
        self.assertEqual(ticket.result.result(timeout=10)['status'],'cancelled')

    def test_http_json_and_sse(self):
        server=make_server(self.worker,port=0)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url=f'http://127.0.0.1:{server.server_port}'
        try:
            with urllib.request.urlopen(url+'/health') as response:
                self.assertEqual(json.load(response)['state'],'ready')
            def request(stream):
                data=json.dumps({'tokens':[1,2,3],'max_new_tokens':4,'stream':stream}).encode()
                with urllib.request.urlopen(urllib.request.Request(url+'/generate',data=data),timeout=10) as response:
                    return response.read().decode()
            result=json.loads(request(False));self.assertEqual(len(result['tokens']),4)
            streamed=request(True);self.assertEqual(streamed.count('event: token'),4)
            self.assertIn('event: done',streamed)
        finally:server.shutdown();server.server_close();thread.join()

    def test_invalid_request_does_not_kill_worker(self):
        ticket=self.worker.submit([-1],4)
        with self.assertRaises(ValueError):ticket.result.result(timeout=10)
        valid=self.worker.submit([1],0).result.result(timeout=10)
        self.assertEqual(valid['tokens'],[])
