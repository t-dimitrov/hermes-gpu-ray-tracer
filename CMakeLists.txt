cmake_minimum_required(VERSION 3.27)
project("Cyclops CPU Ray Tracer")

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES
    "src/Main.cpp"
)

set (HEADER_FILES
    "src/Vec3f.hpp"
    "src/Ray.hpp"
    "src/IHittable.hpp"
    "src/Sphere.hpp"
    "src/Scene.hpp"
    "src/Utility.hpp"
    "src/Interval.hpp"
    "src/Camera.hpp"
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
