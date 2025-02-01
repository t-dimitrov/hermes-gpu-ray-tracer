cmake_minimum_required(VERSION 3.27)
project("Cyclops CPU Ray Tracer")

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES
    "src/Main.cpp"
)

set (HEADER_FILES
    "src/Geometry.hpp"
    "src/Sphere.hpp"
    "src/Material.hpp"
    "src/Ray.hpp"
    "src/Light.hpp"
)

# Create executable
add_executable(Cyclops
    ${SOURCE_FILES}
    ${HEADER_FILES}
)

# Include directories
target_include_directories(Cyclops PRIVATE
    "src/"
)
