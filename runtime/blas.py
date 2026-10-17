"""Small explicit cuBLAS host binding for capture-safe FP32 GEMM.

CuPy's high-level matmul wrapper rejects stream capture on the tested version.
We bind one cuBLAS handle to the execution stream before capture and call SGEMM
without modifying handle state inside the captured region.
"""
import ctypes as ct
from pathlib import Path
import site
import sys


class Blas:
    def __init__(self,stream):
        if sys.platform=='win32':
            candidates=[p for root in site.getsitepackages()
                        for p in Path(root).glob('nvidia/cublas/bin/cublas64_12.dll')]
            self.lib=ct.WinDLL(str(candidates[0]) if candidates else 'cublas64_12.dll')
        else:
            self.lib=ct.CDLL('libcublas.so.12')
        self.handle=ct.c_void_p()
        self.lib.cublasCreate_v2.argtypes=[ct.POINTER(ct.c_void_p)]
        self.lib.cublasSetStream_v2.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.cublasDestroy_v2.argtypes=[ct.c_void_p]
        self.lib.cublasSgemm_v2.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,
            ct.c_int,ct.c_int,ct.c_int,ct.POINTER(ct.c_float),ct.c_void_p,ct.c_int,
            ct.c_void_p,ct.c_int,ct.POINTER(ct.c_float),ct.c_void_p,ct.c_int]
        self._check(self.lib.cublasCreate_v2(ct.byref(self.handle)))
        self._check(self.lib.cublasSetStream_v2(self.handle,ct.c_void_p(stream.ptr)))
        self.alpha,self.beta=ct.c_float(1),ct.c_float(0)

    @staticmethod
    def _check(status):
        if status: raise RuntimeError(f'cuBLAS returned status {status}')

    def linear(self,x,weight,out):
        batch,inner=x.shape;outer=weight.shape[0]
        if weight.shape[1]!=inner or out.shape!=(batch,outer):
            raise ValueError('incompatible GEMM output dimensions')
        # Transpose the row-major result conceptually: C^T = W * X^T.
        status=self.lib.cublasSgemm_v2(self.handle,1,0,outer,batch,inner,
            ct.byref(self.alpha),ct.c_void_p(weight.data.ptr),inner,
            ct.c_void_p(x.data.ptr),inner,ct.byref(self.beta),ct.c_void_p(out.data.ptr),outer)
        self._check(status)

    def close(self):
        if self.handle:
            self._check(self.lib.cublasDestroy_v2(self.handle));self.handle=ct.c_void_p()
