#pragma once
#include <string>
#include <vector>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <Vec3f.hpp>

// Basic usage (see HDR discussion below for HDR usage):
//    int x,y,n;
//    unsigned char *data = stbi_load(filename, &x, &y, &n, 0);
//    // ... process data if not NULL ...
//    // ... x = width, y = height, n = # 8-bit components per pixel ...
//    // ... replace '0' with '1'..'4' to force that many components per pixel
//    // ... but 'n' will always be the number that it would have been if you said 0
//    stbi_image_free(data);

namespace Cyclops
{
    class EnvironmentMap
    {
    public:
        EnvironmentMap(const std::string& filepath)
        {
            int n = -1;
            unsigned char* data = stbi_load(filepath.c_str(), &_width, &_height, &n, 0);

            if (!data || n != 3)
            {
                std::cerr << "Error: Failed to load the environment map" << std::endl;
                return;
            }

            _envMap.resize(_width * _height);
            for (int j = _height - 1; j >= 0; --j)
            {
                for (int i = 0; i < _width; ++i)
                {
                    float x = data[(i + j * _width) * 3 + 0];
                    float y = data[(i + j * _width) * 3 + 1];
                    float z = data[(i + j * _width) * 3 + 2];

                    _envMap[i + j * _width] = Color3f(x, y, z) / 255.0f; // Normalize the color
                }
            }

            stbi_image_free(data);
        }

        ~EnvironmentMap()
        {
        }

        Color3f GetColor(size_t index) const
        {
            assert(index < _envMap.size());
            return _envMap[index];
        }

        int GetWidth() const { return _width; }
        int GetHeight() const { return _height; }

    private:
        std::vector<Color3f> _envMap;
        int _width, _height;
    };
}
