#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "Material.hpp"
#include "Scene.hpp"
#include "Camera.hpp"
#include "Mesh.hpp"

#include "Kernels/GPURayTracer.h"

#include <functional>

using namespace Hermes;

float LinearToGamma2(float linearComponent)
{
    if (linearComponent > 0.0f)
    {
        return std::sqrtf(linearComponent);
    }

    return 0.0f;
}

void SaveTextureToFile(const std::string& filename, const std::vector<Color3f>& texture, int width, int height, float pixelSampleScale)
{
    Interval colorIntensity(0.000f, 0.999f);

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
                // Apply linear to gamma transform for gamma 2
                pixelColor[k] = LinearToGamma2(pixelColor[k] * pixelSampleScale);
                ofs << static_cast<char>(256 * colorIntensity.Clamp(pixelColor[k]));
            }
        }
    }
    ofs.close();
}

int main()
{
    std::shared_ptr<Camera> camera = std::make_shared<Camera>(
        16.0f / 9.0f, 720,
        20.0f, //fov
        Vec3f{ -3.0f, 1.0f, 3.0f }, //look from
        Vec3f{ 0.0f, 0.0f, -1.0f },// look at
        Vec3f{ 0.0f, 1.0f, 0.0f }, // up vector
        100, 100);

    auto scene = std::make_shared<Scene>(
        std::vector<Sphere>{
            Sphere(Point3f(-1.0f, 0.0f, -1.0f), 0.5f, MaterialType::Dielectric, 0),
            Sphere(Point3f(-1.0f, 0.0f, -1.0f), 0.4f, MaterialType::Dielectric, 1),
            Sphere(Point3f(+1.0f, 0.0f, -1.0f), 0.5f, MaterialType::Metal, 0),
        },
        std::vector<Plane>{
            Plane(Point3f(0.0f, -0.5f, -1.0f), Vec3f(0.0f, 1.0f, 0.0f), 5.0f, 5.0f, MaterialType::Lambertian, 0)
        },
        std::vector<Mesh>{
            Mesh("res/duck.obj", Vec3f(-5.0f, 1.3f, 4.0f), 0.2f, MaterialType::Metal, 1)
        },
        std::vector<LambertianMaterial>{
            LambertianMaterial(Color3f(0.8f, 0.8f, 0.0f)),
            LambertianMaterial(Color3f(0.1f, 0.2f, 0.5f))
        },
        std::vector<MetalMaterial>{
            MetalMaterial(Color3f(0.2f, 0.26f, 0.24f), 0.7f),
            MetalMaterial(Color3f(0.8f, 0.6f, 0.2f), 0.1f),
        }, 
        std::vector<DielectricMaterial>{
            DielectricMaterial(1.5f),
            DielectricMaterial(1.0f/1.5f)
        }
    );

    // GPU Ray Tracing
    {
        std::cout << "Width: " << camera->GetImageWidth() << ", Height: " << camera->GetImageHeight() << std::endl;
        Timer cudaRenderTimer("CUDA Render");
        GPURayTracer gpuRT;
        std::vector<Color3f> texture = gpuRT.Render(scene, camera, camera->GetSamplesPerPixel(), camera->GetMaxDepth());

        SaveTextureToFile("./cuda_render.ppm", texture, camera->GetImageWidth(), camera->GetImageHeight(), 1.0f / (float)camera->GetSamplesPerPixel());
    }

    return 0;
}
