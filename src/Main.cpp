#include <iostream>
#include <vector>
#include <fstream>

#include "Geometry.hpp"

void Render()
{
    const int WIDTH = 1024;
    const int HEIGHT = 768;

    std::vector<Vec3f> framebuffer(WIDTH * HEIGHT);

    for (size_t j = 0; j < HEIGHT; ++j)
    {
        for (size_t i = 0; i < WIDTH; ++i)
        {
            framebuffer[i + j * WIDTH] = Vec3f(j / float(HEIGHT), i / float(WIDTH), 0.0f);
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
    std::cout << "Hello Cyclops" << std::endl;

    Render();
    return 0;
}
