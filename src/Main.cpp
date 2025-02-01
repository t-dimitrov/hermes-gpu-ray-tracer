#include <iostream>
#include <vector>
#include <fstream>

#include "Geometry.hpp"
#include "Sphere.hpp"
#include "Ray.hpp"
#include "Light.hpp"

bool SceneIntersect(const Ray& ray, const std::vector<Sphere>& spheres, HitRecord& record)
{
    HitRecord tempRecord;
    constexpr float rayMax = std::numeric_limits<float>::max();
    float closestHit = rayMax;
    bool hitAnything = false;

    for (size_t i = 0; i < spheres.size(); ++i)
    {
        if (spheres[i].Hit(ray, tempRecord))
        {
            hitAnything = true;
            closestHit = tempRecord.t;
            record = tempRecord;
        }
    }

    return hitAnything;
}

Vec3f CastRay(const Ray& ray, const std::vector<Sphere>& spheres, const std::vector<Light>& lights)
{
    HitRecord record;

    if (!SceneIntersect(ray, spheres, record))
    {
        return { 0.2f, 0.7f, 0.8f }; //background color
    }

    // Compute lights
    float diffuseLightIntensity = 0.0f;
    for (size_t i = 0; i < lights.size(); ++i)
    {
        Vec3f lightDir = (lights[i].position - record.point).normalize();
        diffuseLightIntensity += lights[i].intensity * std::max(0.0f, lightDir * record.normal);
    }

    return record.material.diffuse * diffuseLightIntensity;
}

void Render(const std::vector<Sphere>& spheres, const std::vector<Light>& lights)
{
    const int WIDTH = 1024;
    const int HEIGHT = 768;
    const float FOVh = 60.0f; //in degrees

    std::vector<Vec3f> framebuffer(WIDTH * HEIGHT);

    float aspectRatio = static_cast<float>(WIDTH) / static_cast<float>(HEIGHT);
    float scale = std::tan(DegToRad(FOVh) / 2.0f);
    for (size_t j = 0; j < HEIGHT; ++j)
    {
        for (size_t i = 0; i < WIDTH; ++i)
        {
            float x = (2.0f * (i + 0.5f) / (float)WIDTH - 1.0f) * aspectRatio * scale;
            float y = (1.0f - 2.0f * (j + 0.5f) / (float)HEIGHT) * scale;

            Ray ray({ 0.0f, 0.0f, 0.0f }, Vec3f(x, y, -1.0f).normalize());

            framebuffer[i + j * WIDTH] = CastRay(ray, spheres, lights);
        }
    }

    std::ofstream ofs; //save the framebuffer to file
    ofs.open("./out.ppm", std::ofstream::out | std::ios::binary);
    ofs << "P6\n" << WIDTH << " " << HEIGHT << "\n255\n";
    for (size_t i = 0; i < WIDTH*HEIGHT; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            ofs << static_cast<char>(255 * std::max(0.f, std::min(1.f, framebuffer[i][j])));
        }
    }
    ofs.close();
}

int main()
{
    Material ivory(Vec3f(0.4f, 0.4f, 0.3f));
    Material redRubber(Vec3f(0.3f, 0.1f, 0.1f));

    std::vector<Sphere> spheres;
    spheres.emplace_back(Vec3f(-3.0f, 0.0f, -16.0f), 2.0f, ivory);
    spheres.emplace_back(Vec3f(7.0f, 5.0f, -18.0f), 4.0f, ivory);
    spheres.emplace_back(Vec3f(-1.0f, -1.5f, -12.0f), 2.0f, redRubber);
    spheres.emplace_back(Vec3f(1.5f, -0.5f, -18.0f), 3.0f, redRubber);

    std::vector<Light> lights;
    lights.emplace_back(Vec3f(-20.0f, 20.0f, 20.0f), 1.5f);
    
    Render(spheres, lights);
    return 0;
}
