#pragma once

#include "Ray.hpp"
#include "Interval.hpp"

class HitRecord
{
public:
    Point3f point;
    Vec3f normal;
    float t;
    bool frontFace;

    void SetFaceNormal(const Ray& ray, const Vec3f& outwardNormal)
    {
        // Sets the hit record normal vector
        // NOTE: The parameter 'outwardNormal' is assumed to have unit length
        frontFace = Dot(ray.direction(), outwardNormal) < 0.0f;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }
};

class IHittable
{
public:
    virtual ~IHittable() = default;

    virtual bool Hit(const Ray& ray, Interval tRay, HitRecord& hit) const = 0;
};
