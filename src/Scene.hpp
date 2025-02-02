#pragma once
#include <vector>

#include "Sphere.hpp"
#include "Ray.hpp"
#include "Interval.hpp"

class Scene
{
public:
    Scene() {}
    Scene(const std::initializer_list<IHittable*>& list)
        : _hittables(list)
    { }
    ~Scene()
    {
        for (size_t i = 0; i < _hittables.size(); ++i)
        {
            delete _hittables[i];
        }
        _hittables.clear();
    }

    bool Hit(const Ray& ray, Interval tRay, HitRecord& hit)
    {
        HitRecord tempHit;
        bool hitAnything = false;
        float closestHit = tRay.max;

        for (const IHittable* const hittable : _hittables)
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
    std::vector<IHittable*> _hittables;
};
