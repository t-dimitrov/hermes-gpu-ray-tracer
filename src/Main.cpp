#include <iostream>
#include <vector>
#include <fstream>

#include "Utility.hpp"
#include "Vec3f.hpp"
#include "Ray.hpp"
#include "Scene.hpp"
#include "Camera.hpp"

using namespace Cyclops;

int main()
{
    Scene scene({
        std::make_unique<Sphere>(Point3f(0.0f, 0.0f, -1.0f), 0.5f),
        std::make_unique<Sphere>(Point3f(0.0f, -100.5f, -1.0f), 100.0f)
    });

    Camera camera(16.0f/9.0f, 400, 100, 50);
    camera.Render(scene);
    return 0;
}
