#include "Camera.hpp"

#include "Timer.hpp"
#include "Scene.hpp"
#include <fstream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void RenderKernel(Hermes::Vec3f* renderTexture)
{
    int i = threadIdx.x;
    printf("%d\n", i);
}

namespace Hermes
{
    Camera::Camera(float aspectRatio, int imageWidth,
        float FOVv,
        const Point3f& lookFrom, const Point3f& lookAt, const Point3f& up,
        int samplesPerPixel, int maxDepth)
        : _aspectRatio(aspectRatio)
        , _imageWidth(imageWidth)
        , _samplesPerPixel(samplesPerPixel)
        , _maxDepth(maxDepth)
        , _pixelSampleScale(1.0f / _samplesPerPixel)
        , _colorIntensity(0.000f, 0.999f)
        , _FOVv(FOVv)
        , _lookFrom(lookFrom)
        , _lookAt(lookAt)
        , _upVector(up)
    {
        // Calculate image height and ensure it's at least 1
        _imageHeight = (int(_imageWidth / _aspectRatio) < 1) ? 1 : int(_imageWidth / _aspectRatio);

        _center = _lookFrom;

        // Camera
        const float FOCAL_LENGTH = (_lookFrom - _lookAt).Length();

        float theta = DegToRad(_FOVv);
        float h = std::tan(theta / 2.0f);
        const float VP_HEIGHT = 2.0f * h * FOCAL_LENGTH;
        const float VP_WIDTH = VP_HEIGHT * (float(_imageWidth) / _imageHeight);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame
        _w = UnitVector(_lookFrom - _lookAt);
        _u = UnitVector(Cross(_upVector, _w));
        _v = Cross(_w, _u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        Vec3f uViewport = VP_WIDTH * _u; // Vector across viewport horizontal edge
        Vec3f vViewport = VP_HEIGHT * -_v; // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        _uPixelDelta = uViewport / _imageWidth;
        _vPixelDelta = vViewport / _imageHeight;

        // Calculate the location of the upper left pixel.
        Vec3f viewport_upper_left = _center - (FOCAL_LENGTH * _w) - uViewport / 2.0f - vViewport / 2.0f;
        _pixel00Loc = viewport_upper_left + 0.5 * (_uPixelDelta + _vPixelDelta);
    }

    void Camera::Render(const Scene& scene)
    {
        std::vector<Color3f> renderTexture(_imageWidth * _imageHeight);
        Color3f* deviceRT;
        cudaMalloc(&deviceRT, renderTexture.size() * sizeof(Color3f));

        RenderKernel <<<1, 20>>> (deviceRT);
        cudaDeviceSynchronize();
        //return;

        {
            Timer timer("Rendering");
            for (size_t j = 0; j < _imageHeight; ++j)
            {
                for (size_t i = 0; i < _imageWidth; ++i)
                {
                    Color3f& pixelColor = renderTexture[i + j * _imageWidth];
                    for (int sample = 0; sample < _samplesPerPixel; ++sample)
                    {
                        Ray ray = GetRay(i, j);
                        pixelColor += RayColor(ray, _maxDepth, scene);
                    }

                    for (size_t k = 0; k < 3; ++k)
                    {
                        // Apply linear to gamma transform for gamma 2
                        pixelColor[k] = LinearToGamma2(pixelColor[k] * _pixelSampleScale);
                    }
                }
            }
        }

        {
            Timer timer("Saving Render Texture");
            std::ofstream ofs; //save the texture to file
            ofs.open("./out.ppm", std::ofstream::out | std::ios::binary);
            ofs << "P6\n" << _imageWidth << " " << _imageHeight << "\n255\n";
            for (size_t j = 0; j < _imageHeight; ++j)
            {
                for (size_t i = 0; i < _imageWidth; ++i)
                {
                    Color3f pixelColor = renderTexture[i + j * _imageWidth];
                    for (size_t k = 0; k < 3; ++k)
                    {
                        ofs << static_cast<char>(256 * _colorIntensity.Clamp(pixelColor[k]));
                    }
                }
            }
            ofs.close();
        }
    }

    Color3f Camera::RayColor(const Ray& ray, int depth, const Scene& scene)
    {
        // Ray bounce limit has been exceeded.
        if (depth <= 0)
        {
            return Color3f(0.0f, 0.0f, 0.0f);
        }

        HitRecord hit;
        if (scene.Hit(ray, Interval(0.001f, Hermes::INF), hit))
        {
            Ray scattered;
            Color3f attenuation;

            if (hit.material->Scatter(ray, hit, attenuation, scattered))
            {
                return attenuation * RayColor(scattered, depth - 1, scene);
            }

            return Color3f(0.0f, 0.0f, 0.0f);
        }

        // Return background color
        Vec3f unitDir = UnitVector(ray.direction());
        auto a = 0.5f * (unitDir.y() + 1.0f);
        return (1.0f - a) * Color3f(1.0f, 1.0f, 1.0f) + a * Color3f(0.5f, 0.7f, 1.0f);
    }

    // Construct a camera ray originating from the origin and directed 
    // at randomly sampled point around the pixel location (i, j).
    Ray Camera::GetRay(int i, int j) const
    {
        Vec3f offset = SampleSquare();
        Vec3f pixelSample = _pixel00Loc
            + ((i + offset.x()) * _uPixelDelta)
            + ((j + offset.y()) * _vPixelDelta);

        return Ray(_center, pixelSample - _center);
    }

    // Returns the vector to a random point in the [-0.5,-0.5]-[+0.5,+0.5] unit square.
    Vec3f Camera::SampleSquare() const
    {
        return Vec3f(RandomFloat() - 0.5f, RandomFloat() - 0.5f, 0.0f);
    }

    float Camera::LinearToGamma2(float linearComponent)
    {
        if (linearComponent > 0.0f)
        {
            return std::sqrt(linearComponent);
        }

        return 0.0f;
    }
}
