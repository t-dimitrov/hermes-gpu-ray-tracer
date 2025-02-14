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
            int samplesPerPixel, int maxDepth)
            : _aspectRatio(aspectRatio)
            , _imageWidth(imageWidth)
            , _samplesPerPixel(samplesPerPixel)
            , _maxDepth(maxDepth)
            , _pixelSampleScale(1.0f / _samplesPerPixel)
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

        inline int GetImageWidth() const { return _imageWidth; }
        inline int GetImageHeight() const { return _imageHeight; }

        inline int GetSamplesPerPixel() const { return _samplesPerPixel; }
        inline int GetMaxDepth() const { return _maxDepth; }

        inline float GetPixelSampleScale() const { return _pixelSampleScale; }

        // Construct a camera ray originating from the origin and directed 
        // at randomly sampled point around the pixel location (i, j).
        inline Ray GetRay(int i, int j) const
        {
            Vec3f offset = SampleSquare();
            Vec3f pixelSample = _pixel00Loc
                + ((i + offset.x()) * _uPixelDelta)
                + ((j + offset.y()) * _vPixelDelta);

            return Ray(_center, pixelSample - _center);
        }

    private:
        // Returns the vector to a random point in the [-0.5,-0.5]-[+0.5,+0.5] unit square.
        inline Vec3f SampleSquare() const
        {
            return Vec3f(RandomFloat() - 0.5f, RandomFloat() - 0.5f, 0.0f);
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
    };
}
