#pragma once
#include <vector>
#include <fstream>

#include "Camera.hpp"
#include "Sphere.hpp"
#include "Plane.hpp"
#include "Ray.hpp"
#include "Interval.hpp"
#include "Timer.hpp"

namespace Hermes
{
    class Scene
    {
    public:
        Scene() {}
        Scene(const std::initializer_list<Hittable*>& list)
            : _hittables(list)
        {
        }

        ~Scene()
        {
            for (int i = 0; i < _hittables.size(); ++i)
            {
                delete _hittables[i];
            }
        }

        inline bool Hit(const Ray& ray, Interval tRay, HitRecord& hit) const
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
        std::vector<Hittable*> _hittables;
    };
}
