#include "CPURayTracer.hpp"

namespace Hermes
{
    CPURayTracer::CPURayTracer()
    {
    }

    CPURayTracer::~CPURayTracer()
    {
    }

    std::vector<Color3f> CPURayTracer::Render(const std::shared_ptr<Scene>& scene, const std::shared_ptr<Camera>& camera)
    {
        int imageWidth = camera->GetImageWidth();
        int imageHeight = camera->GetImageHeight();
        std::vector<Color3f> renderTexture(imageWidth * imageHeight);

        for (size_t j = 0; j < imageHeight; ++j)
        {
            for (size_t i = 0; i < imageWidth; ++i)
            {
                Color3f& pixelColor = renderTexture[i + j * imageWidth];
                for (int sample = 0; sample < camera->GetSamplesPerPixel(); ++sample)
                {
                    Ray ray = camera->GetRay(i, j);
                    pixelColor += RayColor(ray, camera->GetMaxDepth(), scene);
                }

                for (size_t k = 0; k < 3; ++k)
                {
                    // Apply linear to gamma transform for gamma 2
                    pixelColor[k] = LinearToGamma2(pixelColor[k] * camera->GetPixelSampleScale());
                }
            }
        }

        return renderTexture;
    }

    Color3f CPURayTracer::RayColor(const Ray& ray, int depth, const std::shared_ptr<Scene>& scene)
    {
        // Ray bounce limit has been exceeded.
        if (depth <= 0)
        {
            return Color3f(0.0f, 0.0f, 0.0f);
        }

        HitRecord hit;
        if (scene->Hit(ray, Interval(0.001f, Hermes::INF), hit))
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

    float CPURayTracer::LinearToGamma2(float linearComponent) const
    {
        if (linearComponent > 0.0f)
        {
            return std::sqrt(linearComponent);
        }

        return 0.0f;
    }
}
