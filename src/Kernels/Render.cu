//#include "Camera.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>

__global__ void CudaPrint()
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%d\n", i);
}
