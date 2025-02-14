#pragma once
#include <vector>

#include "Vec3f.hpp"
#include "Scene.hpp"

namespace Hermes
{
    class CPURayTracer
    {
    public:
        CPURayTracer();
        ~CPURayTracer();

        std::vector<Color3f> Render(const std::shared_ptr<Scene>& scene, const std::shared_ptr<Camera>& camera);

        Color3f RayColor(const Ray& ray, int depth, const std::shared_ptr<Scene>& scene);

        float LinearToGamma2(float linearComponent) const;
    };
}