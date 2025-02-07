#pragma once
#include <vector>

#include "Sphere.hpp"
#include "Plane.hpp"
#include "Ray.hpp"
#include "Interval.hpp"
#include "EnvironmentMap.hpp"

namespace Cyclops
{
    class Scene
    {
    public:
        Scene() {}
        Scene(const std::initializer_list<std::shared_ptr<Hittable>>& list)
            : _hittables(list)
        {
        }

        bool Hit(const Ray& ray, Interval tRay, HitRecord& hit) const
        {
            HitRecord tempHit;
            bool hitAnything = false;
            float closestHit = tRay.max;

            for (const auto& hittable : _hittables)
            {
                if (hittable->Hit(ray, Interval(tRay.min, closestHit), tempHit))
                {
                    hitAnything = true;
                    closestHit = tempHit.t;
                    hit = tempHit;
                }
            }

            return hitAnything;
        }

    private:
        std::vector<std::shared_ptr<Hittable>> _hittables;
    };
}
