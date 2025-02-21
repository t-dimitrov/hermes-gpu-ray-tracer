#pragma once
#include "Hittable.hpp"

namespace Hermes
{
    class Plane
    {
    public:
        Plane() {}
        Plane(const Point3f& position, const Vec3f& normal, float width, float height, MaterialType type, int materialId)
            : _position(position)
            , _normal(normal)
            , _width(width)
            , _height(height)
            , _materialType(type)
            , _materialId(materialId)
        {
        }

        __device__ bool HitOnDevice(const Ray& ray, Interval tRay, HitRecord& hit) const
        {
            if (Dot(ray.direction(), _normal) > 0.0f)
            {
                return false;
            }

            float t = Dot(-_normal, (ray.origin() - _position)) / Dot(_normal, ray.direction());
            Vec3f point = ray.At(t);

            bool isWithinX = (point.x() < _position.x() + _width / 2.0f) &&
                (_position.x() - _width / 2.0f < point.x());
            bool isWithinZ = (_position.z() - _height / 2.0f < point.z()) &&
                (point.z() < _position.z() + _height / 2.0f);

            if (!isWithinX || !isWithinZ)
            {
                return false;
            }

            hit.t = t;
            hit.point = ray.At(hit.t);
            hit.SetFaceNormal(ray, _normal);
            hit.materialType = _materialType;
            hit.materialId = _materialId;

            return true;
        }

    private:
        Point3f _position;
        Vec3f _normal;
        float _width, _height;

        MaterialType _materialType;
        int _materialId;
    };
}
