#pragma once
#include <vector>
#include <fstream>

#include "Camera.hpp"
#include "Sphere.hpp"
#include "Plane.hpp"
#include "Mesh.hpp"
#include "Ray.hpp"
#include "Interval.hpp"
#include "Timer.hpp"

#include <cstdio>

namespace Hermes
{
    class Scene
    {
    public:
        Sphere* _spheres;
        uint32_t _sphereCount;

        Plane* _planes;
        uint32_t _planeCount;

        Mesh* _meshes;
        uint32_t _meshCount;
        Vec3f* _meshVertices;
        uint32_t* _meshIndices;
        uint32_t _meshIndexCount;

        LambertianMaterial* _lambertianMaterials;
        uint32_t _lambertianCount;
        MetalMaterial* _metalMaterials;
        uint32_t _metalCount;
        DielectricMaterial* _dielectricMaterials;
        uint32_t _dielectricCount;

    public:
        Scene() {}
        Scene(const std::vector<Sphere>& spheres,
            const std::vector<Plane>& planes,
            const std::vector<Mesh>& meshes,
            const std::vector<LambertianMaterial>& lambertianMaterials,
            const std::vector<MetalMaterial>& metalMaterials,
            const std::vector<DielectricMaterial>& dielectricMaterials)
            : _sphereCount(spheres.size())
            , _planeCount(planes.size())
            , _meshCount(meshes.size())
            , _lambertianCount(lambertianMaterials.size())
            , _metalCount(metalMaterials.size())
            , _dielectricCount(dielectricMaterials.size())
        {
            _spheres = new Sphere[_sphereCount];
            for (int i = 0; i < _sphereCount; ++i)
            {
                _spheres[i] = spheres[i];
            }

            _planes = new Plane[_planeCount];
            for (int i = 0; i < _planeCount; ++i)
            {
                _planes[i] = planes[i];
            }

            _meshes = new Mesh[_meshCount];
            for (int i = 0; i < _meshCount; ++i)
            {
                _meshes[i] = meshes[i];
            }
            _meshIndexCount = _meshes[0]._indices.size();

            _lambertianMaterials = new LambertianMaterial[_lambertianCount];
            for (int i = 0; i < _lambertianCount; ++i)
            {
                _lambertianMaterials[i] = lambertianMaterials[i];
            }

            _metalMaterials = new MetalMaterial[_metalCount];
            for (int i = 0; i < _metalCount; ++i)
            {
                _metalMaterials[i] = metalMaterials[i];
            }

            _dielectricMaterials = new DielectricMaterial[_dielectricCount];
            for (int i = 0; i < _dielectricCount; ++i)
            {
                _dielectricMaterials[i] = dielectricMaterials[i];
            }
        }

        ~Scene()
        {
            delete[] _spheres;
            delete[] _planes;
            delete[] _meshes;

            delete[] _lambertianMaterials;
            delete[] _metalMaterials;
            delete[] _dielectricMaterials;
        }


        __device__ inline bool DidHit(const Ray& ray, Interval tRay, HitRecord& hit)
        {
            HitRecord tempHit;
            bool hitAnything = false;
            float closestHit = tRay.max;

            for (int i = 0; i < _sphereCount; ++i)
            {
                if (_spheres[i].HitOnDevice(ray, Interval(tRay.min, closestHit), tempHit))
                {
                    if (tempHit.t < closestHit)
                    {
                        hitAnything = true;
                        closestHit = tempHit.t;
                        hit = tempHit;
                    }
                }
            }

            for (int i = 0; i < _planeCount; ++i)
            {
                if (_planes[i].HitOnDevice(ray, Interval(tRay.min, closestHit), tempHit))
                {
                    if (tempHit.t < closestHit)
                    {
                        hitAnything = true;
                        closestHit = tempHit.t;
                        hit = tempHit;
                    }
                }
            }

            for (int i = 0; i < _meshCount; ++i)
            {
                if (_meshes[i].HitOnDevice(ray, Interval(tRay.min, closestHit), tempHit, _meshVertices, _meshIndices, _meshIndexCount))
                {
                    if (tempHit.t < closestHit)
                    {
                        hitAnything = true;
                        closestHit = tempHit.t;
                        hit = tempHit;
                    }
                }
            }

            return hitAnything;
        }

        __device__ inline bool ScatterLambertian(curandState state, int materialId, const Ray& rayIn, const HitRecord& hit, Color3f& attenuation, Ray& scattered) const
        {
            if (materialId >= _lambertianCount)
            {
                printf("Invalid lambertian materialId\n");
                return false;
            }

            return _lambertianMaterials[materialId].Scatter(state, rayIn, hit, attenuation, scattered);
        }

        __device__ inline bool ScatterMetal(curandState state, int materialId, const Ray& rayIn, const HitRecord& hit, Color3f& attenuation, Ray& scattered) const
        {
            if (materialId >= _metalCount)
            {
                printf("Invalid metal materialId\n");
                return false;
            }

            return _metalMaterials[materialId].Scatter(state, rayIn, hit, attenuation, scattered);
        }

        __device__ inline bool ScatterDielectric(curandState state, int materialId, const Ray& rayIn, const HitRecord& hit, Color3f& attenuation, Ray& scattered) const
        {
            if (materialId >= _dielectricCount)
            {
                printf("Invalid dielectric materialId\n");
                return false;
            }

            return _dielectricMaterials[materialId].Scatter(state, rayIn, hit, attenuation, scattered);
        }
    };
}
