#pragma once
#include "Geometry.hpp"

struct Ray
{
    Vec3f origin;
    Vec3f direction;

    Ray(const Vec3f& origin, const Vec3f& dir)
        : origin(origin)
        , direction(dir)
    { }
    Ray() {}

    Vec3f At(float t) const
    {
        return origin + direction * t;
    }
};
