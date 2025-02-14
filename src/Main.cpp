#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "Material.hpp"
#include "Scene.hpp"
#include "Camera.hpp"
#include "Mesh.hpp"

#include "CPURayTracer.hpp"
#include "Kernels/GPURayTracer.h"

#include <functional>

using namespace Hermes;

void SaveTextureToFile(const std::string& filename, const std::vector<Color3f>& texture, int width, int height)
{
    Interval colorIntensity(0.000f, 0.999f); // Translates the [0,1] component values to the byte range [0, 255]

    std::ofstream ofs; //save the texture to file
    ofs.open(filename.c_str(), std::ofstream::out | std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (size_t j = 0; j < height; ++j)
    {
        for (size_t i = 0; i < width; ++i)
        {
            Color3f pixelColor = texture[i + j * width];
            for (size_t k = 0; k < 3; ++k)
            {
                ofs << static_cast<char>(256 * colorIntensity.Clamp(pixelColor[k]));
            }
        }
    }
    ofs.close();
}

int main()
{
    std::shared_ptr<Material> groundMat = std::make_shared<LambertianMaterial>(Color3f(0.8f, 0.8f, 0.0f));
    std::shared_ptr<Material> leftMat   = std::make_shared<DielectricMaterial>(1.5f);
    std::shared_ptr<Material> bubbleMat = std::make_shared<DielectricMaterial>(1.0f/1.5f);
    std::shared_ptr<Material> rightMat = std::make_shared<MetalMaterial>(Color3f(0.2f, 0.26f, 0.24f), 0.7f);
    std::shared_ptr<Material> duckMat  = std::make_shared<MetalMaterial>(Color3f(0.8f, 0.6f, 0.2f), 0.1f);

    std::shared_ptr<Camera> camera = std::make_shared<Camera>(
        16.0f / 9.0f, 400,
        20.0f, //fov
        Vec3f{ -3.0f, 1.0f, 3.0f }, //look from
        Vec3f{ 0.0f, 0.0f, -1.0f },// look at
        Vec3f{ 0.0f, 1.0f, 0.0f }, // up vector
        1, 50);

    auto scene = std::make_shared<Scene>(std::initializer_list<Hittable*>{ 
        new Plane(Point3f(0.0f, -0.5f, -1.0f), Vec3f(0.0f, 1.0f, 0.0f), 5.0f, 5.0f, groundMat), 
        new Sphere(Point3f(-1.0f, 0.0f, -1.0f), 0.5f, leftMat),
        new Sphere(Point3f(-1.0f, 0.0f, -1.0f), 0.4f, bubbleMat),
        new Sphere(Point3f(+1.0f, 0.0f, -1.0f), 0.5f, rightMat),
        new Mesh("res/duck.obj", Vec3f(-5.0f, 1.3f, 4.0f), 0.2f, duckMat)
    });

    // CPU Ray Tracing
    #if false
    {
        Timer cpuRenderTimer("CPU Render");
        CPURayTracer cpuRT;
        std::vector<Color3f> texture = cpuRT.Render(scene, camera);

        SaveTextureToFile("./cpu_render.ppm", texture, camera->GetImageWidth(), camera->GetImageHeight());
    }
    #endif

    // GPU Ray Tracing
    {
        Timer cudaRenderTimer("CUDA Render");
        GPURayTracer gpuRT;
        std::vector<Color3f> texture = gpuRT.Render(scene, camera);

        SaveTextureToFile("./cuda_render.ppm", texture, camera->GetImageWidth(), camera->GetImageHeight());
    }

    return 0;
}
