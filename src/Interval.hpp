#pragma once
#include "Utility.hpp"

namespace Hermes
{
    class Interval
    {
    public:
        float min, max;

        //Default interval is empty, therefore min and max are swapped
        __host__ __device__ Interval()
            : min(+Hermes::INF)
            , max(-Hermes::INF)
        {
        }

        __host__ __device__ Interval(float min, float max)
            : min(min)
            , max(max)
        {
        }

        __host__ __device__ float Size() const { return max - min; }
        __host__ __device__ bool Contains(float x) const { return min <= x && x <= max; }
        __host__ __device__ bool Surrounds(float x) const { return min < x && x < max; }
        __host__ __device__ float Clamp(float x) const
        {
            if (x < min) return min;
            if (x > max) return max;
            return x;
        }

        static const Interval s_empty;
        static const Interval s_universe;
    };
}

inline const Hermes::Interval Hermes::Interval::s_empty = Hermes::Interval(+Hermes::INF, -Hermes::INF);
inline const Hermes::Interval Hermes::Interval::s_universe = Hermes::Interval(-Hermes::INF, +Hermes::INF);
