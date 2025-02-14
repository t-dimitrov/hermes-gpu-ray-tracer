#pragma once
#include <vector>

#include "Vec3f.hpp"
#include "Scene.hpp"

namespace Hermes
{
    class GPURayTracer
    {
    public:
        GPURayTracer();
        ~GPURayTracer();

        __host__ std::vector<Color3f> Render(const std::shared_ptr<Scene>& scene, const std::shared_ptr<Camera>& camera);

    private:
        void CheckDevices() const;

        void InitDeviceMemory();
        void ReleaseDeviceMemory();

    private:
        std::shared_ptr<Scene> _scene;
        std::shared_ptr<Camera> _camera;

        Color3f* _deviceRT;
    };
}
