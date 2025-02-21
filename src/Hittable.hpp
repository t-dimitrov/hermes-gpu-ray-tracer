#pragma once

#include "Ray.hpp"
#include "Interval.hpp"

#include <cuda_runtime.h>

namespace Hermes
{
    class Material;
    enum class MaterialType;
    class HitRecord
    {
    public:
        Point3f point;
        Vec3f normal;
        float t;
        bool frontFace;

        MaterialType materialType;
        int materialId;

        __device__ inline void SetFaceNormal(const Ray& ray, const Vec3f& outwardNormal)
        {
            // Sets the hit record normal vector
            // NOTE: The parameter 'outwardNormal' is assumed to have unit length
            frontFace = Dot(ray.direction(), outwardNormal) < 0.0f;
            normal = frontFace ? outwardNormal : -outwardNormal;
        }
    };
}
