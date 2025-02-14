#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "Hittable.hpp"
#include "Vec3f.hpp"

namespace Hermes
{
    class Mesh : public Hittable
    {
    public:
        Mesh(const std::string& filename, const Vec3f& position, float scale, const std::shared_ptr<Material>& material)
            : _position(position)
            , _scale(scale)
            , _aabbMin(INF, INF, INF)
            , _aabbMax(-INF, -INF, -INF)
            , _material(material)
        {
            LoadObj(filename);

            // Update position and scale of Mesh's AABB
            for (size_t i = 0; i < 3; ++i)
            {
                _aabbMin.e[i] = (_aabbMin.e[i] + _position.e[i]) * _scale;
                _aabbMax.e[i] = (_aabbMax.e[i] + _position.e[i]) * _scale;
            }
        }

        ~Mesh()
        {

        }

        /*
        Implemented using this Geometric test:
        */
        bool Hit(const Ray& ray, Interval tRay, HitRecord& hit) const override
        {
            if (!HitAABB(ray, tRay))
            {
                return false;
            }

            bool hitAnything = false;
            float closestHit = std::numeric_limits<float>::max();
            for (int index = 0; index < _indices.size(); index += 3)
            {
                Vec3f v0 = (_vertices[_indices[index + 0]-1] + _position) * _scale;
                Vec3f v1 = (_vertices[_indices[index + 1]-1] + _position) * _scale;
                Vec3f v2 = (_vertices[_indices[index + 2]-1] + _position) * _scale;

                // Counter-clock wise
                Vec3f v0v1 = v1 - v0;
                Vec3f v0v2 = v2 - v0;

                // No need to normalize yet
                Vec3f N = Cross(v0v1, v0v2);

                // Check if ray and plane are parallel
                float NdotRayDirection = Dot(N, ray.direction());
                if (std::fabs(NdotRayDirection) < 0.0001f)
                {
                    continue;
                }

                float d = -(Dot(N, v0));
                float t = -(Dot(N, ray.origin()) + d) / NdotRayDirection;

                // Check if the triangle is behind the ray
                if (!(tRay.Surrounds(t)))
                {
                    continue;
                }

                Vec3f P = ray.At(t);

                // Inside-out test
                Vec3f Ne; // Vector perpendicular to triangle's plane

                // Test sidedness of P w.r.t. edge v0v1
                Vec3f v0p = P - v0;
                Ne = Cross(v0v1, v0p);

                // Check if P is on the right side
                if (Dot(N, Ne) < 0.0f)
                {
                    continue;
                }

                // Test sidedness of P w.r.t. edge v2v1
                Vec3f v2v1 = v2 - v1;
                Vec3f v1p = P - v1;
                Ne = Cross(v2v1, v1p);

                // Check if P is on the right side
                if (Dot(N, Ne) < 0.0f)
                {
                    continue;
                }

                // Test sidedness of P w.r.t. edge v2v0
                Vec3f v2v0 = v0 - v2;
                Vec3f v2p = P - v2;
                Ne = Cross(v2v0, v2p);

                // Check if P is on the right side
                if (Dot(N, Ne) < 0.0f)
                {
                    continue;
                }

                // Consider this hit point
                if (t < closestHit)
                {
                    closestHit = t;

                    hit.t = t;
                    hit.point = P;
                    hit.SetFaceNormal(ray, Normalize(N));
                    hit.material = _material;
                    hitAnything = true;
                }
            }

            return hitAnything;
        }

    private:
        bool LoadObj(const std::string& filename)
        {
            std::ifstream ifs(filename.c_str());
            if (!ifs.is_open())
            {
                std::cerr << "Error: failed to open " << filename << std::endl;
                return false;
            }
            
            std::string line;
            while (std::getline(ifs, line))
            {
                std::istringstream ss(line);
                char token;
                ss >> token;
                
                // Process vertices
                if (token == 'v' || token == 'V')
                {
                    float x, y, z;
                    ss >> x >> y >> z;
                    _vertices.emplace_back(x, y, z);

                    UpdateBoundingBox({ x, y, z });
                }

                // Process faces
                if (token == 'f' || token == 'F')
                {
                    uint32_t v0, v1, v2; //indices of vertices that form a face
                    ss >> v0 >> v1 >> v2;
                    _indices.emplace_back(v0);
                    _indices.emplace_back(v1);
                    _indices.emplace_back(v2);
                }
            }

            ifs.close();
            return true;
        }

        void UpdateBoundingBox(const Vec3f& v)
        {
            for (size_t i = 0; i < 3; ++i)
            {
                if (v[i] < _aabbMin.e[i])
                {
                    _aabbMin.e[i] = v[i];
                }

                if (v[i] > _aabbMax.e[i])
                {
                    _aabbMax.e[i] = v[i];
                }
            }
        }

        bool HitAABB(const Ray& ray, Interval tRay) const
        {
            float t1 = (_aabbMin.x() - ray.origin().x()) / ray.direction().x();
            float t2 = (_aabbMax.x() - ray.origin().x()) / ray.direction().x();
            float t3 = (_aabbMin.y() - ray.origin().y()) / ray.direction().y();
            float t4 = (_aabbMax.y() - ray.origin().y()) / ray.direction().y();
            float t5 = (_aabbMin.z() - ray.origin().z()) / ray.direction().z();
            float t6 = (_aabbMax.z() - ray.origin().z()) / ray.direction().z();
            float t7 = std::fmax(std::fmax(std::fmin(t1, t2), std::fmin(t3, t4)), std::fmin(t5, t6));
            float t8 = std::fmin(std::fmin(std::fmax(t1, t2), std::fmax(t3, t4)), std::fmax(t5, t6));

            if (t8 < 0.0f || t7 > t8)
            {
                return false;
            }

            return true;
        }

    private:
        Vec3f _position;
        float _scale;

        Vec3f _aabbMin;
        Vec3f _aabbMax;

        std::vector<Vec3f> _vertices;
        std::vector<uint32_t> _indices;
        std::shared_ptr<Material> _material;
    };
}
