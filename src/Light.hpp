#pragma once
#include "Geometry.hpp"

struct Light
{
    Vec3f position;
    float intensity;

    Light(const Vec3f& position, float intensity)
        : position(position)
        , intensity(intensity)
    { }
};
