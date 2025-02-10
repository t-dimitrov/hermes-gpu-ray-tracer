#include <iostream>
#include <vector>
#include <fstream>

#include "Material.hpp"
#include "Scene.hpp"
#include "Camera.hpp"
#include "Model.hpp"

#include <functional>

using namespace Hermes;

int main()
{
    std::shared_ptr<Material> groundMat = std::make_shared<LambertianMaterial>(Color3f(0.8f, 0.8f, 0.0f));
    std::shared_ptr<Material> leftMat   = std::make_shared<DielectricMaterial>(1.5f);
    std::shared_ptr<Material> bubbleMat = std::make_shared<DielectricMaterial>(1.0f/1.5f);
    std::shared_ptr<Material> rightMat = std::make_shared<MetalMaterial>(Color3f(0.2f, 0.26f, 0.24f), 0.7f);
    std::shared_ptr<Material> duckMat  = std::make_shared<MetalMaterial>(Color3f(0.8f, 0.6f, 0.2f), 0.1f);

    Scene scene({
        std::make_unique<Plane>(Point3f(0.0f, -0.5f, -1.0f), Vec3f(0.0f, 1.0f, 0.0f), 5.0f, 5.0f, groundMat),
        //std::make_unique<Sphere>(Point3f( 0.0f,  0.0f,   -1.2f),   0.5f, centerMat),
        std::make_unique<Sphere>(Point3f(-1.0f,  0.0f,   -1.0f),   0.5f, leftMat),
        std::make_unique<Sphere>(Point3f(-1.0f,  0.0f,   -1.0f),   0.4f, bubbleMat),
        std::make_unique<Sphere>(Point3f(+1.0f,  0.0f,   -1.0f),   0.5f, rightMat),
        std::make_shared<Model>("res/duck.obj", Vec3f(-5.0f, 1.3f, 4.0f), 0.2f, duckMat),
    });

    Camera camera(
        16.0f / 9.0f, 400,
        20.0f, //fov
        { -3.0f, 1.0f, 3.0f }, //look from
        { 0.0f, 0.0f, -1.0f },// look at
        { 0.0f, 1.0f, 0.0f }, // up vector
        1, 50);

    camera.Render(scene);

    return 0;
}
