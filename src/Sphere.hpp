#pragma once
#include "Geometry.hpp"
#include "Material.hpp"
#include "Ray.hpp"
#include "HitRecord.hpp"

struct Sphere
{
    Vec3f center;
    float radius;
    Material material;

    Sphere(const Vec3f& center, const float& radius, const Material& mat)
        : center(center)
        , radius(radius)
        , material(mat)
    { }

    bool Hit(const Ray& ray, HitRecord& record) const
    {
        // https://gamemath.com/book/geomtests.html#intersection_ray_sphere
        Vec3f e = center - ray.origin;
        float a = e * ray.direction;
        float b2 = e * e - a * a;

        float f = sqrt(radius * radius - e * e + a * a);

        if (std::isnan(f) || f < 0.0f)
        {
            return false;
        }

        if (e * e < radius * radius) // ray is inside the sphere
        {
            return false;
        }

        float t = a - f;

        // Set hit record
        if (t < record.t)
        {
            record.t = t;
            record.material = material;
            record.point = ray.At(t);
            record.normal = (record.point - center).normalize();
            return true;
        }

        return false;
    }
};
