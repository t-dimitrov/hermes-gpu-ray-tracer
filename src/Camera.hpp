#pragma once

#include "Hittable.hpp"
#include "Material.hpp"

namespace Cyclops
{
    /*
    The camera class is responsible for two important jobs:
        1. Construct and dispatch rays into the scene
        2. Use the results of these rays to construct the render texture
    */
    class Camera
    {
    public:
        Camera(float aspectRatio, int imageWidth, 
            float FOVv, 
            const Point3f& lookFrom, const Point3f& lookAt, const Point3f& up,
            int samplesPerPixel, int maxDepth)
            : _aspectRatio(aspectRatio)
            , _imageWidth(imageWidth)
            , _samplesPerPixel(samplesPerPixel)
            , _maxDepth(maxDepth)
            , _pixelSampleScale(1.0f/_samplesPerPixel)
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

        void Render(const Scene& scene)
        {
            std::ofstream ofs; //save the texture to file
            ofs.open("./out.ppm", std::ofstream::out | std::ios::binary);
            ofs << "P6\n" << _imageWidth << " " << _imageHeight << "\n255\n";
            for (size_t j = 0; j < _imageHeight; ++j)
            {
                std::cout << "\rScanlines remaining: " << j + 1 << "/" << _imageHeight << ' ' << std::flush;
                for (size_t i = 0; i < _imageWidth; ++i)
                {
                    Color3f pixelColor;
                    for (int sample = 0; sample < _samplesPerPixel; ++sample)
                    {
                        Ray ray = GetRay(i, j);
                        pixelColor += RayColor(ray, _maxDepth, scene);
                    }

                    for (size_t k = 0; k < 3; ++k)
                    {
                        // Apply linear to gamma transform for gamma 2
                        pixelColor[k] = LinearToGamma2(pixelColor[k] * _pixelSampleScale);

                        ofs << static_cast<char>(256 * _colorIntensity.Clamp(pixelColor[k]));
                    }
                }
            }
            ofs.close();
            std::cout << "\rDone.                       " << std::endl;
        }

    private:
        Color3f RayColor(const Ray& ray, int depth, const Scene& scene)
        {
            // Ray bounce limit has been exceeded.
            if (depth <= 0)
            {
                return Color3f(0.0f, 0.0f, 0.0f);
            }

            HitRecord hit;
            if (scene.Hit(ray, Interval(0.001f, Cyclops::INF), hit))
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
        Ray GetRay(int i, int j) const
        {
            Vec3f offset = SampleSquare();
            Vec3f pixelSample = _pixel00Loc 
                + ((i + offset.x()) * _uPixelDelta) 
                + ((j + offset.y()) * _vPixelDelta);

            return Ray(_center, pixelSample - _center);
        }

        // Returns the vector to a random point in the [-0.5,-0.5]-[+0.5,+0.5] unit square.
        Vec3f SampleSquare() const
        {
            return Vec3f(RandomFloat() - 0.5f, RandomFloat() - 0.5f, 0.0f);
        }

        float LinearToGamma2(float linearComponent)
        {
            if (linearComponent > 0.0f)
            {
                return std::sqrt(linearComponent);
            }

            return 0.0f;
        }

    private:
        float _aspectRatio; // Ratio of image width over height
        int _samplesPerPixel; // Count of random samples for each pixel
        float _pixelSampleScale; // Color scale factor for a sum of pixel samples
        int _maxDepth; // Maximum number of ray bounces into the scene
        int _imageWidth; // Rendered image width in pixels
        int _imageHeight; // Rendered image height int pixels
        
        float _FOVv; // Vertical view angle (field of view)
        Point3f _lookFrom; // Point camera is looking from
        Point3f _lookAt; // Point camera is looking at
        Vec3f _upVector; // Camera-relative "up" direction
        
        Vec3f _u, _v, _w; // Camera frame basis vectors

        Point3f _center; // Camera center position
        Point3f _pixel00Loc; // Location of pixel (0, 0)
        Vec3f _uPixelDelta; // Offset to pixel to the right
        Vec3f _vPixelDelta; // Offset to pixel below

        const Interval _colorIntensity; // Translates the [0,1] component values to the byte range [0, 255]
    };
}
