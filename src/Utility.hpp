#pragma once

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

#include "Vec3f.hpp"

namespace Cyclops
{
    // Constants
    constexpr float INF = std::numeric_limits<float>::max();
    constexpr float PI = 3.1415926535897932385;

    // Utility func
    inline float DegToRad(float degrees)
    {
        return degrees * PI / 180.0f;
    }

    class Random
    {
    public:
        static float Float(float min, float max)
        {
            static std::random_device device;
            static std::mt19937 generator(device());
            std::uniform_real_distribution<float> dist(min, max);
            return dist(generator);
        }

        static float Float()
        {
            static std::random_device device;
            static std::mt19937 generator(device());
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            return dist(generator);
        }
    };
}
