#pragma once
#include <vector>
#include <curand_kernel.h>

#include "Vec3f.hpp"
#include "Scene.hpp"

namespace Hermes
{
    class GPURayTracer
    {
    public:
        GPURayTracer();
        ~GPURayTracer();

        __host__ std::vector<Color3f> Render(const std::shared_ptr<Scene>& scene, const std::shared_ptr<Camera>& camera, int samplesPerPixel, int depth);

    private:
        void CheckDevices() const;

        void InitDeviceMemory();
        void ReleaseDeviceMemory();

    private:
        std::shared_ptr<Scene> _scene;
        std::shared_ptr<Camera> _camera;

        curandState* _curandState;
        Color3f* _deviceRT;
        Camera* _deviceCamera;

        Scene* _deviceScene;
        Sphere* _tempSpheres;
        Plane* _tempPlanes;
        Mesh* _tempMeshes;
        Vec3f* _tempMeshVertices;
        uint32_t* _tempMeshIndices;

        LambertianMaterial* _tempLambertianMats;
        MetalMaterial* _tempMetalMats;
        DielectricMaterial* _tempDielectricMats;
    };
}
