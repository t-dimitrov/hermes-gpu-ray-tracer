cmake_minimum_required(VERSION 3.27)
project("Cyclops CPU Ray Tracer")

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES
    "src/Main.cpp"
)

set (HEADER_FILES
    "src/Vec3f.hpp"
    "src/Ray.hpp"
    "src/Hittable.hpp"
    "src/Sphere.hpp"
    "src/Scene.hpp"
    "src/Plane.hpp"
    "src/Utility.hpp"
    "src/Interval.hpp"
    "src/Camera.hpp"
    "src/Material.hpp"
    "src/Timer.hpp"
    "src/EnvironmentMap.hpp"

    "src/stb_image.h"
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

# Copy 'res' folder to output folder
add_custom_target(copy_res_folder ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${PROJECT_SOURCE_DIR}/res
    ${PROJECT_BINARY_DIR}/res
    COMMENT "Copying 'res' folder into binary directory"
)

add_dependencies(Cyclops copy_res_folder)
