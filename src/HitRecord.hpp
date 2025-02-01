#pragma once
#include "Geometry.hpp"
#include "Material.hpp"

struct HitRecord
{
    float t;
    Vec3f point;
    Vec3f normal;
    Material material;

    HitRecord()
        : t(std::numeric_limits<float>::max())
        , point(0.0f, 0.0f, 0.0f)
        , normal(0.0f, 0.0f, 0.0f)
        , material()
    { }
};
