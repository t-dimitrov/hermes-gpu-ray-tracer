#pragma once
#include "Utility.hpp"

class Interval
{
public:
    float min, max;

    //Default interval is empty, therefore min and max are swapped
    Interval()
        : min(+Cyclops::INF)
        , max(-Cyclops::INF)
    { }

    Interval(float min, float max)
        : min(min)
        , max(max)
    { }

    float Size() const { return max - min; }
    bool Contains(float x) const { return min <= x && x <= max; }
    bool Surrounds(float x) const { return min < x && x < max; }

    static const Interval s_empty;
    static const Interval s_universe;
};

const Interval Interval::s_empty = Interval(+Cyclops::INF, -Cyclops::INF);
const Interval Interval::s_universe = Interval(-Cyclops::INF, +Cyclops::INF);
