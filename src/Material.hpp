#pragma once

#include "Hittable.hpp"
#include "Utility.hpp"

namespace Hermes
{
    /*
    The Material class is responsible for two things:
        1. Produce a scattered ray (or say it absorbed the incident ray)
        2. If scattered, say how much of the ray should be attenuated
    */
    class Material
    {
    public:
        virtual ~Material() = default;

        virtual bool Scatter(const Ray& rayIn, const HitRecord& hit, Color3f& attenuation, Ray& scattered) const
        {
            return false;
        }
    };

    class LambertianMaterial : public Material
    {
    public:
        LambertianMaterial(const Color3f& albedo)
            : _albedo(albedo) {}

        bool Scatter(const Ray& rayIn, const HitRecord& hit, Color3f& attenuation, Ray& scattered) const override
        {
            Vec3f scatterDir = hit.normal + RandomUnitVec3f();

            // Catch degenerate scatter dirs
            if (scatterDir.IsNearZero())
            {
                scatterDir = hit.normal;
            }

            scattered = Ray(hit.point, scatterDir);
            attenuation = _albedo;

            return true;
        }

    private:
        Color3f _albedo;
    };

    class MetalMaterial : public Material
    {
    public:
        MetalMaterial(const Color3f& albedo, float fuzz)
            : _albedo(albedo)
            , _fuzz(fuzz < 1.0f ? fuzz : 1.0f)
        {}

        bool Scatter(const Ray& rayIn, const HitRecord& hit, Color3f& attenuation, Ray& scattered) const override
        {
            Vec3f reflected = Reflect(rayIn.direction(), hit.normal);
            reflected = UnitVector(reflected) + (_fuzz * RandomUnitVec3f());

            scattered = Ray(hit.point, reflected);
            attenuation = _albedo;

            return (Dot(scattered.direction(), hit.normal) > 0.0f);
        }

    private:
        Color3f _albedo;
        float _fuzz;
    };

    class DielectricMaterial : public Material
    {
    public:
        DielectricMaterial(float refractionIndex) 
            : _refractionIndex(refractionIndex)
        {}

        bool Scatter(const Ray& rayIn, const HitRecord& hit, Color3f& attenuation, Ray& scattered) const override
        {
            attenuation = Color3f(1.0f, 1.0f, 1.0f);
            float ri = hit.frontFace ? (1.0f / _refractionIndex) : _refractionIndex;

            Vec3f unitDir = UnitVector(rayIn.direction());
            float cosTheta = std::fmin(Dot(-unitDir, hit.normal), 1.0f);
            float sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);

            bool cannotRefract = ri * sinTheta > 1.0f;
            Vec3f direction;
            if (cannotRefract || Reflectance(cosTheta, ri) > RandomFloat())
            {
                direction = Reflect(unitDir, hit.normal);
            }
            else
            {
                direction = Refract(unitDir, hit.normal, ri);
            }

            scattered = Ray(hit.point, direction);
            return true;
        }

    private:
        // Use Schlick's approximation for reflectance
        float Reflectance(float cosine, float refractionIndex) const
        {
            float r0 = (1.0f - refractionIndex) / (1.0f + refractionIndex);
            r0 = r0 * r0;
            return r0 + (1.0f - r0) * std::pow((1.0f - cosine), 5);
        }

    private:
        // Refractive index in vacuum or air, or the ratio of the material's refractive index over
        // the refractive index of the enclosing media
        double _refractionIndex;
    };
}
