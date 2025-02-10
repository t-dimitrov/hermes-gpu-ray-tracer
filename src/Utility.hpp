#pragma once

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

#include "Vec3f.hpp"

namespace Hermes
{
    // Constants
    constexpr float INF = std::numeric_limits<float>::max();
    constexpr float PI = 3.1415926535897932385;

    // Utility func
    inline float DegToRad(float degrees)
    {
        return degrees * PI / 180.0f;
    }

    static float RandomFloat(float min, float max)
    {
        static std::uniform_real_distribution<float> dist(min, max);
        static std::mt19937 generator;
        return dist(generator);
    }

    static float RandomFloat()
    {
        static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        static std::mt19937 generator;
        return dist(generator);
    }

    static Vec3f RandomVec3f(float min, float max)
    {
        return Vec3f(RandomFloat(min, max), RandomFloat(min, max), RandomFloat(min, max));
    }

    static Vec3f RandomVec3f()
    {
        return Vec3f(RandomFloat(), RandomFloat(), RandomFloat());
    }

    static Vec3f RandomUnitVec3f()
    {
        return Normalize(RandomVec3f(-1.0f, 1.0f));
    }

    static Vec3f RandomOnHemisphere(const Vec3f& normal) 
    {
        Vec3f onUnitSphere = RandomUnitVec3f();
        if (Dot(onUnitSphere, normal) > 0.0) // In the same hemisphere as the normal
        {
            return onUnitSphere;
        }

        return -onUnitSphere;
    }
}
