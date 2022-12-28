cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)
project(KinectFusion_remake)

# Options
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Libs CACHE PATH "Path to lib folder")
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src CACHE PATH "Path to source folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
set(glog_DIR ${LIBRARY_DIR}/glog-lib/lib/cmake/glog CACHE PATH "Path to installed glog")
set(Ceres_DIR ${LIBRARY_DIR}/Ceres/CMake CACHE PATH "Path to installed Ceres")
set(Flann_INCLUDE_DIR ${LIBRARY_DIR}/Flann-1.8.4/ CACHE PATH "Path to Flann source folder")

# Use pre-compiled libraries on Windows
if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x64/ CACHE PATH "Path to FreeImage header file")
        set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x64/ CACHE PATH "Path to FreeImage .lib/.dll folder")
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x32/ CACHE PATH "Path to FreeImage header file")
        set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x32/ CACHE PATH "Path to FreeImage .lib/.dll folder")
    endif()
endif(WIN32)

# Set C++ flags

#find_package(nanoflann REQUIRED)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED)

find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)

include_directories("${CUDA_INCLUDE_DIRS}")
enable_language(CUDA)
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_22,code=sm_22
    )


set(HEADER_FILES
    ${SOURCE_DIR}/SurfaceMeasurement.cuh
    ${SOURCE_DIR}/PointCloud.h
    ${SOURCE_DIR}/VirtualSensor.h
    ${SOURCE_DIR}/PointCloud.h
    ${SOURCE_DIR}/FreeImageHelper.h
    ${SOURCE_DIR}/Eigen.h
    ${SOURCE_DIR}/SimpleMesh.h



)

set(SOURCE_FILES 
    ${SOURCE_DIR}/FreeImageHelper.cpp

    ${SOURCE_DIR}/SurfaceMeasurement.cu
)

link_directories(${FreeImage_LIBRARY_DIR})
add_executable(KinectFusion_remake main.cpp ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(KinectFusion_remake ceres freeimage Eigen3::Eigen )
target_compile_options(KinectFusion_remake PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --expt-relaxed-constexpr
                       >)

target_include_directories(KinectFusion_remake PUBLIC ${EIGEN3_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR} ${Flann_INCLUDE_DIR})