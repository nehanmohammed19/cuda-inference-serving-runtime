"""CUDA device ownership, ordered queues, pinned staging and event retirement.

Unlike a software work queue, CommandQueue enqueues real GPU commands. It does
not start a background Python worker. Objects retain their owning device runtime.
"""
from collections import OrderedDict
import numpy as np


class CommandQueue:
    def __init__(self, cp, name):
        self.cp, self.name = cp, name
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.submissions = 0

    def record(self):
        event = self.cp.cuda.Event(disable_timing=True)
        event.record(self.stream)
        return event

    def wait(self, event):
        self.stream.wait_event(event)

    def finish(self):
        self.stream.synchronize()


class RetirementQueue:
    """Keep resources alive until the event protecting their last use completes."""
    def __init__(self):
        self.pending = []

    def retire(self, event, callback, keepalive=None):
        self.pending.append((event, callback, keepalive))

    def collect(self, wait=False):
        remaining = []
        for event, callback, keepalive in self.pending:
            if wait:
                event.synchronize()
            if wait or event.query():
                callback()
            else:
                remaining.append((event, callback, keepalive))
        self.pending = remaining


class Workspace:
    """Named buffers whose addresses stay stable throughout program lifetime."""
    def __init__(self, xp, budget_bytes=None):
        self.xp, self.budget_bytes = xp, budget_bytes
        self.buffers = {}
        self.bytes = 0
        self.sealed = False

    def allocate(self, name, shape, dtype=np.float32):
        if name in self.buffers:
            old = self.buffers[name]
            if old.shape != tuple(shape) or old.dtype != np.dtype(dtype):
                raise ValueError('workspace signature changed: ' + name)
            return old
        if self.sealed:
            raise RuntimeError('cannot allocate after workspace sealing')
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if self.budget_bytes is not None and self.bytes + size > self.budget_bytes:
            raise MemoryError('workspace budget exceeded')
        array = self.xp.empty(shape, dtype=dtype)
        self.buffers[name] = array
        self.bytes += size
        return array

    def seal(self):
        self.sealed = True

    def report(self):
        return {'bytes': self.bytes, 'sealed': self.sealed,
                'buffers': {name: {'shape': list(a.shape), 'bytes': a.nbytes}
                            for name, a in self.buffers.items()}}


class ProgramCache:
    """Bounded LRU: compilation state is reused, dynamic runtime arguments are not."""
    def __init__(self, capacity=4):
        if capacity < 1:
            raise ValueError('program capacity must be positive')
        self.capacity = capacity
        self.entries = OrderedDict()
        self.hits = self.misses = self.evictions = 0

    def get_or_create(self, signature, factory):
        if signature in self.entries:
            self.hits += 1
            self.entries.move_to_end(signature)
            return self.entries[signature]
        self.misses += 1
        # Compile before evicting: a failed factory leaves the cache usable.
        program = factory()
        if len(self.entries) >= self.capacity:
            _, old = self.entries.popitem(last=False)
            old.close()  # waits for the program's last-use event
            self.evictions += 1
        self.entries[signature] = program
        return program

    def close(self):
        for value in self.entries.values(): value.close()
        self.entries.clear()

    def report(self):
        return {'entries': len(self.entries), 'hits': self.hits, 'misses': self.misses,
                'evictions': self.evictions,
                'workspace_bytes': sum(p.workspace.bytes for p in self.entries.values())}


class StagingBuffer:
    def __init__(self, cp, shape, dtype):
        self.owner = cp.cuda.alloc_pinned_memory(int(np.prod(shape))*np.dtype(dtype).itemsize)
        self.host = np.frombuffer(self.owner, dtype=dtype, count=int(np.prod(shape))).reshape(shape)
        self.device = cp.empty(shape, dtype=dtype)
        self.last_upload = None

    def upload(self, value, queue):
        # DMA still owns pinned memory until this event completes. Waiting here
        # prevents CPU writes racing a transfer from an earlier invocation.
        if self.last_upload is not None: self.last_upload.synchronize()
        np.copyto(self.host, value)
        self.device.set(self.host, stream=queue.stream)
        self.last_upload = queue.record()
        queue.submissions += 1
        return self.last_upload


class DeviceRuntime:
    def __init__(self, backend):
        if backend.device != 'cuda': raise ValueError('device runtime requires CUDA')
        self.backend, self.cp = backend, backend.xp
        self.compute = CommandQueue(self.cp, 'compute')
        self.transfer = CommandQueue(self.cp, 'transfer')
        self.retirement = RetirementQueue()

    def close(self):
        self.compute.finish(); self.transfer.finish()
        self.retirement.collect(wait=True)
