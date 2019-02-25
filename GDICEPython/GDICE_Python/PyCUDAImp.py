import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


# 1 block per sample
# nSim threads per block
multinomialTMod = SourceModule("""
  __global__ void sampleT(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2;
  } 
""")