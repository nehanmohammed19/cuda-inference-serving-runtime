"""Device operations, with all custom kernels in kernels/ops.cu."""
from pathlib import Path
import os
import sys
import numpy as np
from . import reference


class Backend:
    def __init__(self, device='cpu'):
        self.device = device
        self._dlls = []
        if device == 'cpu':
            self.xp = np
            return
        if device != 'cuda': raise ValueError('device must be cpu or cuda')
        # CUDA pip wheels avoid requiring a system-wide toolkit installation.
        if sys.platform == 'win32':
            import site
            for root in site.getsitepackages():
                for path in Path(root).glob('nvidia/*/bin'):
                    self._dlls.append(os.add_dll_directory(str(path)))
                    os.environ['PATH'] = str(path) + os.pathsep + os.environ['PATH']
        import cupy as cp
        self.xp = cp
        cp.cuda.Device().use()
        source = (Path(__file__).parents[1] / 'kernels/ops.cu').read_text()
        self.module = cp.RawModule(code=source, options=('--std=c++11',))
        self.rms_kernel = self.module.get_function('rms')
        self.rope_kernel = self.module.get_function('rotate')
        self.attention_kernel = self.module.get_function('paged_attention')
        self.split_kernel = self.module.get_function('split_attention')
        self.merge_kernel = self.module.get_function('merge_attention')

    def sync(self):
        if self.device == 'cuda': self.xp.cuda.get_current_stream().synchronize()

    def numpy(self, x):
        return x if self.device == 'cpu' else self.xp.asnumpy(x)

    def rmsnorm(self, x, weight, eps=1e-5):
        if self.device == 'cpu': return reference.rmsnorm(x, weight, eps)
        out = self.xp.empty_like(x)
        self.rms_kernel((x.shape[0],), (256,),
                        (x, weight, out, np.int32(x.shape[-1]), np.float32(eps)))
        return out

    def rope(self, x, positions, theta=10000.):
        if self.device == 'cpu': return reference.rope(x, positions, theta)
        b, h, d = x.shape
        if d % 2: raise ValueError('RoPE requires even head dimension')
        out = self.xp.empty_like(x)
        self.rope_kernel(((h*d//2+255)//256, b), (256,),
                         (x, positions, out, np.int32(h), np.int32(d), np.float32(theta)))
        return out

    def attention(self, q, cache, keys):
        if self.device == 'cpu':
            return np.stack([reference.attention(q[i], *cache.dense(key))
                             for i, key in enumerate(keys)])
        tables, lengths = cache.metadata(keys)
        out = self.xp.empty_like(q)
        splits=min(16,(max(cache.lengths[k] for k in keys)+127)//128)
        if splits>1:
            partial,stats=self.split_workspace(q,splits)
            self.launch_split(q,cache,tables,lengths,out,partial,stats,splits)
        else:
            self.launch_attention(q, cache, tables, lengths, out)
        return out

    def split_workspace(self,q,splits):
        b,h,d=q.shape
        return (self.xp.empty((b,h,splits,d),dtype=self.xp.float32),
                self.xp.empty((b,h,splits,2),dtype=self.xp.float32))

    def launch_split(self,q,cache,tables,lengths,out,partial,stats,splits):
        b,h,d=q.shape
        if d>256 or splits<1 or h%cache.k.shape[2]: raise ValueError('unsupported split geometry')
        self.split_kernel((b,h,splits),(256,),
            (q,cache.k,cache.v,tables,lengths,partial,stats,np.int32(tables.shape[1]),
             np.int32(cache.page_size),np.int32(h),np.int32(cache.k.shape[2]),np.int32(d),np.int32(splits)))
        self.merge_kernel((b,h),(256,),(partial,stats,out,np.int32(h),np.int32(d),np.int32(splits)))

    def launch_attention(self, q, cache, tables, lengths, out):
        b, h, d = q.shape
        if d > 256 or h % cache.k.shape[2]: raise ValueError('unsupported head geometry')
        self.attention_kernel((b, h), (256,),
            (q, cache.k, cache.v, tables, lengths, out, np.int32(tables.shape[1]),
             np.int32(cache.page_size), np.int32(h), np.int32(cache.k.shape[2]), np.int32(d)))

    def capture_attention(self, q, cache, tables, lengths, out):
        """Fixed pointers/shapes only. Caller keeps buffers alive and updates contents.

        This intentionally does not claim whole-model graph capture: model.step
        allocates intermediates and prepares host cache metadata each iteration.
        """
        if self.device != 'cuda': raise ValueError('graphs require CUDA')
        stream = self.xp.cuda.Stream(non_blocking=True)
        self.sync()
        with stream:
            self.launch_attention(q, cache, tables, lengths, out)
            stream.synchronize()
            stream.begin_capture()
            self.launch_attention(q, cache, tables, lengths, out)
            graph = stream.end_capture()
        stream.synchronize()
        return graph
