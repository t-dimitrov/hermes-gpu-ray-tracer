#pragma once
#include "Vec3f.hpp"

namespace Cyclops
{
    class Ray
    {
    public:
        Ray() {}
        Ray(const Vec3f& origin, const Vec3f& direction)
            : _origin(origin)
            , _direction(direction)
        {
        }

        Vec3f At(float t) const
        {
            return _origin + _direction * t;
        }

        const Vec3f origin() const { return _origin; }
        const Vec3f direction() const { return _direction; }

    private:
        Vec3f _origin;
        Vec3f _direction;
    };
}
