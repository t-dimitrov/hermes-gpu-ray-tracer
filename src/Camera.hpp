#pragma once

#include "Hittable.hpp"
#include "Material.hpp"

#include <cuda_runtime.h>

namespace Hermes
{
    class Scene;
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
            int samplesPerPixel, int maxDepth);

        void Render(const Scene& scene);

    private:

    private:
        Color3f RayColor(const Ray& ray, int depth, const Scene& scene);

        // Construct a camera ray originating from the origin and directed 
        // at randomly sampled point around the pixel location (i, j).
        Ray GetRay(int i, int j) const;

        // Returns the vector to a random point in the [-0.5,-0.5]-[+0.5,+0.5] unit square.
        Vec3f SampleSquare() const;

        float LinearToGamma2(float linearComponent);

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
