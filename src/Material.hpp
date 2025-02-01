#pragma once
#include "Geometry.hpp"

struct Material
{
    Vec3f diffuse;

    Material(const Vec3f& color)
        : diffuse(color)
    { }

    Material()
        : diffuse(0.0f, 0.0f, 0.0f)
    { }
};
