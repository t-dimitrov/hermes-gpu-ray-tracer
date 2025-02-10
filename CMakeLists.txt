cmake_minimum_required(VERSION 3.29)

project("Hermes GPU Ray Tracer" LANGUAGES CXX)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES
    "src/Main.cpp"
    
    # CUDA src files
    "src/Kernels/Render.cu"
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
    "src/Model.hpp"
    "src/EnvironmentMap.hpp"

    "src/stb_image.h"
)

# Create executable
add_executable(Hermes
    ${SOURCE_FILES}
    ${HEADER_FILES}
)

# Include directories
target_include_directories(Hermes PRIVATE
    "src/"
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set_target_properties(Hermes PROPERTIES
    CMAKE_CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Compile for the architecture(s) of the host's GPU(s).
#set_target_properties(Hermes PROPERTIES CUDA_ARCHITECTURES native)

# Copy 'res' folder to output folder
add_custom_target(copy_res_folder ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${PROJECT_SOURCE_DIR}/res
    ${PROJECT_BINARY_DIR}/res
    COMMENT "Copying 'res' folder into binary directory"
)

add_dependencies(Hermes copy_res_folder)
