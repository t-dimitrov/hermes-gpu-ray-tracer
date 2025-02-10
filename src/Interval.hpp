#pragma once
#include "Utility.hpp"

namespace Hermes
{
    class Interval
    {
    public:
        float min, max;

        //Default interval is empty, therefore min and max are swapped
        Interval()
            : min(+Hermes::INF)
            , max(-Hermes::INF)
        {
        }

        Interval(float min, float max)
            : min(min)
            , max(max)
        {
        }

        float Size() const { return max - min; }
        bool Contains(float x) const { return min <= x && x <= max; }
        bool Surrounds(float x) const { return min < x && x < max; }
        float Clamp(float x) const
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
