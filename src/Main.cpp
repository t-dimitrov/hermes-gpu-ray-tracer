#include <iostream>
#include <vector>
#include <fstream>

#include "Utility.hpp"
#include "Vec3f.hpp"
#include "Ray.hpp"
#include "Scene.hpp"
#include "Camera.hpp"
#include "Timer.hpp"
#include "EnvironmentMap.hpp"

using namespace Cyclops;

int main()
{
    auto envMap = std::make_shared<EnvironmentMap>("res/envmap.jpg");

    std::shared_ptr<Material> groundMat = std::make_shared<LambertianMaterial>(Color3f(0.8f, 0.8f, 0.0f));
    std::shared_ptr<Material> centerMat = std::make_shared<LambertianMaterial>(Color3f(0.1f, 0.2f, 0.5f));
    std::shared_ptr<Material> leftMat   = std::make_shared<DielectricMaterial>(1.5f);
    std::shared_ptr<Material> bubbleMat = std::make_shared<DielectricMaterial>(1.0f/1.5f);
    std::shared_ptr<Material> rightMat  = std::make_shared<MetalMaterial>(Color3f(0.8f, 0.6f, 0.2f), 0.0f);

    Scene scene({
        //std::make_unique<Sphere>(Point3f( 0.0f, -100.5f, -1.0f), 100.0f, groundMat),
        std::make_unique<Plane>(Point3f(0.0f, -0.5f, -1.0f), Vec3f(0.0f, 1.0f, 0.0f), 5, 5, groundMat),
        std::make_unique<Sphere>(Point3f( 0.0f,  0.0f,   -1.2f),   0.5f, centerMat),
        std::make_unique<Sphere>(Point3f(-1.0f,  0.0f,   -1.0f),   0.5f, leftMat),
        std::make_unique<Sphere>(Point3f(-1.0f,  0.0f,   -1.0f),   0.4f, bubbleMat),
        std::make_unique<Sphere>(Point3f(+1.0f,  0.0f,   -1.0f),   0.5f, rightMat),
    }, envMap);

    Camera camera(
        16.0f / 9.0f, 400,
        20.0f, //fov
        { 2.0f, 1.0f, 5.0f }, //look from
        { 0.0f, 0.0f, -1.0f },// look at
        { 0.0f, 1.0f, 0.0f }, // up vector
        100, 50);

    Timer timer("Main Render");
    camera.Render(scene);

    return 0;
}
