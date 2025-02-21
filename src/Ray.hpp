#pragma once
#include "Vec3f.hpp"

#include <cuda_runtime.h>

namespace Hermes
{
    class Ray
    {
    public:
        __host__ __device__ Ray() {}
        __host__ __device__ Ray(const Vec3f& origin, const Vec3f& direction)
            : _origin(origin)
            , _direction(direction)
        {
        }

        __host__ __device__ Vec3f At(float t) const
        {
            return _origin + _direction * t;
        }

        __host__ __device__ const Vec3f origin() const { return _origin; }
        __host__ __device__ const Vec3f direction() const { return _direction; }

    private:
        Vec3f _origin;
        Vec3f _direction;
    };
}
