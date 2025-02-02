#pragma once

#include "IHittable.hpp"

namespace Cyclops
{
    class Sphere : public IHittable
    {
    public:
        Sphere(const Point3f& center, float radius)
            : _center(center)
            , _radius(radius)
        {
        }

        bool Hit(const Ray& ray, Interval tRay, HitRecord& hit) const override
        {
            Vec3f oc = _center - ray.origin();
            auto a = ray.direction().LengthSquared();
            auto h = Dot(ray.direction(), oc);
            auto c = oc.LengthSquared() - _radius * _radius;

            auto discriminant = h * h - a * c;
            if (discriminant < 0.0f)
            {
                return false;
            }

            float sqrtd = std::sqrt(discriminant);

            // Find the nearest root that lies between [tMin, tMax]
            float root = (h - sqrtd) / a;
            if (!tRay.Surrounds(root))
            {
                root = (h + sqrtd) / a;
                if (!tRay.Surrounds(root))
                {
                    return false;
                }
            }

            hit.t = root;
            hit.point = ray.At(root);
            Vec3f outwardNormal = (hit.point - _center) / _radius;
            hit.SetFaceNormal(ray, outwardNormal);

            return true;
        }

    private:
        Point3f _center;
        float _radius;
    };
}
