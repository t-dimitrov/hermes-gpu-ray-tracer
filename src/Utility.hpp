#pragma once

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>

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
}
