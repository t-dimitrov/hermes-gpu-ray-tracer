#include "GPURayTracer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>

namespace Hermes
{
    __global__ void RenderKernel(Color3f* renderTexture, int width, int height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        // Check if within bounds
        if (x > width || y > height)
        {
            return;
        }

        int index = x + y * width;
        renderTexture[index] = Color3f((float)x/width, (float)y/height, 0.0f);
    }

    GPURayTracer::GPURayTracer()
        : _scene(nullptr)
        , _camera(nullptr)
    {
        CheckDevices();

        // Choose which GPU to run on, change this on a multi-GPU system.
        cudaError_t cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaSetDevice failed! Do you have a CUDA-capable GPU installed? (" << cudaStatus << ")" << std::endl;
            return;
        }
        std::cout << "Running on device 0. To change go to 'Kernels/GPURayTracer.cu'" << std::endl;
    }

    GPURayTracer::~GPURayTracer()
    {
    }

    void GPURayTracer::CheckDevices() const
    {
        int deviceCount;
        cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaGetDeviceCount failed (" << cudaStatus << ")" << std::endl;
            return;
        }

        std::cout << "CUDA found " << deviceCount << " device(s)" << std::endl;
        for (int i = 0; i < deviceCount; ++i)
        {
            cudaDeviceProp props;
            cudaStatus = cudaGetDeviceProperties_v2(&props, i);
            if (cudaStatus != cudaSuccess)
            {
                std::cerr << "cudaGetDeviceProperties_v2 failed (" << cudaStatus << ")" << std::endl;
                return;
            }

            std::cout << "Device Number: " << i << std::endl;
            std::cout << "\tDevice name: " << props.name << std::endl;
            std::cout << "\tMemory Clock Rate: " << props.memoryClockRate / 1024 << std::endl;
            std::cout << "\tMemory Bus Width (bits): " << props.memoryBusWidth << std::endl;
            std::cout << "\tPeak Memory Bandwidth (GB/s): " << 2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1.0e6 << std::endl;
            std::cout << "\tTotal global memory (Gbytes): " << (float)(props.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0 << std::endl;
            std::cout << "\tShared memory per block (Kbytes): " << (float)(props.sharedMemPerBlock) / 1024.0 << std::endl;
            std::cout << "\tMinor-Major: " << props.minor << "-" << props.major << std::endl;
            std::cout << "\tWarp size: " << props.warpSize << std::endl;

            const char* concurrentKernels = props.concurrentKernels ? "yes" : "no";
            std::cout << "\tConcurrent kernels: " << concurrentKernels << std::endl;

            const char* concurrentComputation = props.deviceOverlap ? "yes" : "no";
            std::cout << "\tConcurrent computation/communication: " << concurrentComputation << std::endl;
        }
    }

    void GPURayTracer::InitDeviceMemory()
    {
        int imageWidth = _camera->GetImageWidth();
        int imageHeight = _camera->GetImageHeight();

        cudaError_t cudaStatus = cudaMalloc((void**)&_deviceRT, imageWidth * imageHeight * sizeof(Color3f));
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaMalloc failed for deviceRT (" << cudaStatus << ")" << std::endl;
        }
    }

    void GPURayTracer::ReleaseDeviceMemory()
    {
        cudaFree(_deviceRT);
    }

    std::vector<Color3f> GPURayTracer::Render(const std::shared_ptr<Scene>& scene, const std::shared_ptr<Camera>& camera)
    {
        _scene = scene;
        _camera = camera;
        InitDeviceMemory();

        int imageWidth = camera->GetImageWidth();
        int imageHeight = camera->GetImageHeight();
        std::vector<Color3f> renderTexture(imageWidth * imageHeight);

        dim3 threadsPerBlock(1, 1); // TODO: Will be samples per pixel
        dim3 blocksPerGrid((imageWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (imageHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

        RenderKernel<<<blocksPerGrid, threadsPerBlock>>>(_deviceRT, imageWidth, imageHeight);
        cudaDeviceSynchronize();

        cudaError_t cudaStatus = cudaMemcpy((void*)renderTexture.data(), (void*)_deviceRT, imageWidth * imageHeight * sizeof(Vec3f), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaMemcpy failed for deviceRT to host (" << cudaStatus << ")" << std::endl;
        }

        ReleaseDeviceMemory();
        return renderTexture;
    }
}
