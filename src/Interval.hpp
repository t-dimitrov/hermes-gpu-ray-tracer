#pragma once
#include "Utility.hpp"

namespace Cyclops
{
    class Interval
    {
    public:
        float min, max;

        //Default interval is empty, therefore min and max are swapped
        Interval()
            : min(+Cyclops::INF)
            , max(-Cyclops::INF)
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

const Cyclops::Interval Cyclops::Interval::s_empty = Cyclops::Interval(+Cyclops::INF, -Cyclops::INF);
const Cyclops::Interval Cyclops::Interval::s_universe = Cyclops::Interval(-Cyclops::INF, +Cyclops::INF);
