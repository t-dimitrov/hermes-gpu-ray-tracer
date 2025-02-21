#include "GPURayTracer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>

namespace Hermes
{
    __device__ Color3f RayColor(curandState state, const Ray& ray, int depth, Scene* scene)
    {
        Color3f color(1.0f, 1.0f, 1.0f);
        Ray currentRay = ray;

        for (int i = 0; i < depth; ++i)
        {
            HitRecord hit;
            if (scene->DidHit(currentRay, Interval(0.001f, 1000.0f), hit))
            {
                Ray scattered;
                Color3f attenuation;

                if (hit.materialType == MaterialType::Lambertian)
                {
                    if (scene->ScatterLambertian(state, hit.materialId, currentRay, hit, attenuation, scattered))
                    {
                        currentRay = scattered;
                        color = color * attenuation;
                    }
                    else
                    {
                        return { 0.0f, 0.0f, 0.0f };
                    }
                }
                else if (hit.materialType == MaterialType::Metal)
                {
                    if (scene->ScatterMetal(state, hit.materialId, currentRay, hit, attenuation, scattered))
                    {
                        currentRay = scattered;
                        color = color * attenuation;
                    }
                    else
                    {
                        return { 0.0f, 0.0f, 0.0f };
                    }
                }
                else if (hit.materialType == MaterialType::Dielectric)
                {
                    if (scene->ScatterDielectric(state, hit.materialId, currentRay, hit, attenuation, scattered))
                    {
                        currentRay = scattered;
                        color = color * attenuation;
                    }
                    else
                    {
                        return { 0.0f, 0.0f, 0.0f };
                    }
                }
                else
                {
                    // Return magenta when no material is found
                    return { 1.0f, 0.0f, 1.0f };
                }
            }
            else
            {
                // Background color
                Vec3f unitDir = UnitVector(currentRay.direction());
                auto a = 0.5f * (unitDir.y() + 1.0f);
                Color3f bgColor = (1.0f - a) * Color3f(1.0f, 1.0f, 1.0f) + a * Color3f(0.5f, 0.7f, 1.0f);
                return color * bgColor;
            }
        }

        // Ray bounce limit has been exceeded.
        return color * 0.75f; //{ 0.0f, 0.0f, 0.0f };
    }

    __global__ void RenderKernel(Color3f* renderTexture, int width, int height, int samplesPerPixel, int depth, Camera* camera, Scene* scene, unsigned long long seed)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        // Check if within bounds
        if (x > width || y > height)
        {
            return;
        }

        // Init curand
        curandState state;
        curand_init(seed, x + y * width, 0, &state);

        Color3f color(0.0f, 0.0f, 0.0f);
        for (int sample = 0; sample < samplesPerPixel; ++sample)
        {
            Vec3f sampleSquare(curand_uniform(&state) - 0.5f, curand_uniform(&state) - 0.5f, 0.0f);
            Ray ray = camera->GetRayDevice(x, y, sampleSquare);
            color += RayColor(state, ray, depth, scene);
        }

        int index = x + y * width;
        renderTexture[index] = color;
    }

    GPURayTracer::GPURayTracer()
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

        // Init curand
        CheckCuda(cudaMalloc((void**)&_curandState, imageWidth * sizeof(curandState)));

        // Init render texture
        CheckCuda(cudaMalloc((void**)&_deviceRT, imageWidth * imageHeight * sizeof(Color3f)));

        // Init device camera
        CheckCuda(cudaMalloc((void**)&_deviceCamera, sizeof(Camera)));

        CheckCuda(cudaMemcpy((void*)_deviceCamera, (void*)_camera.get(), sizeof(Camera), cudaMemcpyHostToDevice));

        // Init device scene
        CheckCuda(cudaMalloc((void**)&_deviceScene, sizeof(Scene)));
        CheckCuda(cudaMemcpy(_deviceScene, _scene.get(), sizeof(Scene), cudaMemcpyHostToDevice));

        // Malloc temp shapes arrays
        CheckCuda(cudaMalloc((void**)&_tempSpheres, sizeof(Sphere) * _scene->_sphereCount));
        CheckCuda(cudaMemcpy(&(_deviceScene->_spheres), &_tempSpheres, sizeof(Sphere*), cudaMemcpyHostToDevice));

        CheckCuda(cudaMalloc((void**)&_tempPlanes, sizeof(Plane) * _scene->_planeCount));
        CheckCuda(cudaMemcpy(&(_deviceScene->_planes), &_tempPlanes, sizeof(Plane*), cudaMemcpyHostToDevice));

        CheckCuda(cudaMalloc((void**)&_tempMeshes, sizeof(Mesh) * _scene->_meshCount));
        CheckCuda(cudaMemcpy(&(_deviceScene->_meshes), &_tempMeshes, sizeof(Mesh*), cudaMemcpyHostToDevice))

        // Copy shapes data
        CheckCuda(cudaMemcpy(_tempSpheres, _scene->_spheres, sizeof(Sphere) * _scene->_sphereCount, cudaMemcpyHostToDevice));
        CheckCuda(cudaMemcpy(_tempPlanes, _scene->_planes, sizeof(Plane) * _scene->_planeCount, cudaMemcpyHostToDevice));
        CheckCuda(cudaMemcpy(_tempMeshes, _scene->_meshes, sizeof(Mesh) * _scene->_meshCount, cudaMemcpyHostToDevice));

        // Malloc temp mesh data
        CheckCuda(cudaMalloc((void**)&_tempMeshVertices, sizeof(Vec3f) * _scene->_meshes[0]._vertices.size()));
        CheckCuda(cudaMemcpy(&(_deviceScene->_meshVertices), &_tempMeshVertices, sizeof(Vec3f*), cudaMemcpyHostToDevice));

        CheckCuda(cudaMalloc((void**)&_tempMeshIndices, sizeof(uint32_t) * _scene->_meshes[0]._indices.size()));
        CheckCuda(cudaMemcpy(&(_deviceScene->_meshIndices), &_tempMeshIndices, sizeof(uint32_t*), cudaMemcpyHostToDevice));

        // Copy temp mesh data
        CheckCuda(cudaMemcpy(_tempMeshVertices, _scene->_meshes[0]._vertices.data(), sizeof(Vec3f) * _scene->_meshes[0]._vertices.size(), cudaMemcpyHostToDevice));
        CheckCuda(cudaMemcpy(_tempMeshIndices, _scene->_meshes[0]._indices.data(), sizeof(uint32_t) * _scene->_meshes[0]._indices.size(), cudaMemcpyHostToDevice));

        // Malloc temp materials arrays
        CheckCuda(cudaMalloc((void**)&_tempLambertianMats, sizeof(LambertianMaterial) * _scene->_lambertianCount));
        CheckCuda(cudaMemcpy(&(_deviceScene->_lambertianMaterials), &_tempLambertianMats, sizeof(LambertianMaterial*), cudaMemcpyHostToDevice));

        CheckCuda(cudaMalloc((void**)&_tempMetalMats, sizeof(MetalMaterial) * _scene->_metalCount));
        CheckCuda(cudaMemcpy(&(_deviceScene->_metalMaterials), &_tempMetalMats, sizeof(MetalMaterial*), cudaMemcpyHostToDevice));

        CheckCuda(cudaMalloc((void**)&_tempDielectricMats, sizeof(DielectricMaterial) * _scene->_dielectricCount));
        CheckCuda(cudaMemcpy(&(_deviceScene->_dielectricMaterials), &_tempDielectricMats, sizeof(DielectricMaterial*), cudaMemcpyHostToDevice));

        // Copy temp material arrays data
        CheckCuda(cudaMemcpy(_tempLambertianMats, _scene->_lambertianMaterials, sizeof(LambertianMaterial) * _scene->_lambertianCount, cudaMemcpyHostToDevice));
        CheckCuda(cudaMemcpy(_tempMetalMats, _scene->_metalMaterials, sizeof(MetalMaterial) * _scene->_metalCount, cudaMemcpyHostToDevice));
        CheckCuda(cudaMemcpy(_tempDielectricMats, _scene->_dielectricMaterials, sizeof(DielectricMaterial) * _scene->_dielectricCount, cudaMemcpyHostToDevice));
    }

    void GPURayTracer::ReleaseDeviceMemory()
    {
        CheckCuda(cudaFree(_tempLambertianMats));
        CheckCuda(cudaFree(_tempMetalMats));
        CheckCuda(cudaFree(_tempDielectricMats));

        CheckCuda(cudaFree(_tempMeshes));
        CheckCuda(cudaFree(_tempPlanes));
        CheckCuda(cudaFree(_tempSpheres));

        CheckCuda(cudaFree(_deviceScene));
        CheckCuda(cudaFree(_deviceCamera));
        CheckCuda(cudaFree(_deviceRT));
        CheckCuda(cudaFree(_curandState));
    }

    std::vector<Color3f> GPURayTracer::Render(const std::shared_ptr<Scene>& scene, const std::shared_ptr<Camera>& camera, int samplesPerPixel, int depth)
    {
        _scene = scene;
        _camera = camera;
        cudaError_t cudaStatus;
        InitDeviceMemory();

        int imageWidth = camera->GetImageWidth();
        int imageHeight = camera->GetImageHeight();
        std::vector<Color3f> renderTexture(imageWidth * imageHeight, Vec3f(0.0f, 0.0f, 0.0f));
        cudaStatus = cudaMemcpy((void*)_deviceRT, (void*)renderTexture.data(), imageWidth * imageHeight * sizeof(Vec3f), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaMemcpy failed for deviceRT to host (" << cudaStatus << ")" << std::endl;
        }

        dim3 threadsPerBlock(1, 1);
        dim3 blocksPerGrid((imageWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (imageHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

        RenderKernel<<<blocksPerGrid, threadsPerBlock>>>(_deviceRT, imageWidth, imageHeight, samplesPerPixel, depth,
            _deviceCamera, _deviceScene, 42);
        cudaDeviceSynchronize();

        CheckCuda(cudaMemcpy((void*)renderTexture.data(), (void*)_deviceRT, imageWidth * imageHeight * sizeof(Vec3f), cudaMemcpyDeviceToHost));

        ReleaseDeviceMemory();
        return renderTexture;
    }
}
