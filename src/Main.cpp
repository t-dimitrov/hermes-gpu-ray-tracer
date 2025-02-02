#include <iostream>
#include <vector>
#include <fstream>

#include "Vec3f.hpp"
#include "Ray.hpp"

Color3f RayColor(const Ray& ray)
{
    Vec3f unitDir = UnitVector(ray.direction());
    auto a = 0.5f * (unitDir.y() + 1.0f);
    return (1.0f - a) * Color3f(1.0f, 1.0f, 1.0f) + a * Color3f(0.5f, 0.7f, 1.0f);
}

void Render()
{
    // Image
    constexpr float ASPECT_RATIO = 16.0f / 9.0f;
    constexpr int IMAGE_WIDTH = 400;
    // Calculate image height and ensure it's at least 1
    constexpr int IMAGE_HEIGHT = (int(IMAGE_WIDTH / ASPECT_RATIO) < 1) ? 1 : int(IMAGE_WIDTH / ASPECT_RATIO);

    // Canera
    constexpr float FOCAL_LENGTH = 1.0f;
    constexpr float VP_HEIGHT = 2.0f;
    constexpr float VP_WIDTH = VP_HEIGHT * (float(IMAGE_WIDTH) / IMAGE_HEIGHT);
    Point3f cameraCenter = { 0.0f, 0.0f, 0.0f };

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    Vec3f uViewport = Vec3f(VP_WIDTH, 0.0f, 0.0f);
    Vec3f vViewport = Vec3f(0.0f, -VP_HEIGHT, 0.0f);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    Vec3f uPixelDelta = uViewport / IMAGE_WIDTH;
    Vec3f vPixelDelta = vViewport / IMAGE_HEIGHT;

    // Calculate the location of the upper left pixel.
    Vec3f viewport_upper_left = cameraCenter - Vec3f(0.0f, 0.0f, FOCAL_LENGTH) - uViewport / 2.0f - vViewport / 2.0f;
    Vec3f pixel00_loc = viewport_upper_left + 0.5 * (uPixelDelta + vPixelDelta);

    // Render
    std::ofstream ofs; //save the framebuffer to file
    ofs.open("./out.ppm", std::ofstream::out | std::ios::binary);
    ofs << "P6\n" << IMAGE_WIDTH << " " << IMAGE_HEIGHT << "\n255\n";
    for (size_t j = 0; j < IMAGE_HEIGHT; ++j)
    {
        std::cout << "\rScanlines remaining: " << j+1 << "/" << IMAGE_HEIGHT << ' ' << std::flush;
        for (size_t i = 0; i < IMAGE_WIDTH; ++i)
        {
            Vec3f pixelCenter = pixel00_loc + (i * uPixelDelta) + (j * vPixelDelta);
            Vec3f rayDirection = pixelCenter - cameraCenter;
            Ray ray(cameraCenter, rayDirection);

            Color3f pixelColor = RayColor(ray);

            for (size_t k = 0; k < 3; ++k)
            {
                ofs << static_cast<char>(255.999f * pixelColor[k]);
            }
        }
    }
    ofs.close();

    std::cout << "\rDone.                       " << std::endl;
}

int main()
{
    Render();
    return 0;
}
