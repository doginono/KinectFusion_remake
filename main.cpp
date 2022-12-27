#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "src/SurfaceMeasurement.cuh"
#include <iostream>
#include "src/Eigen.h"
#include <vector>
#include <string>
#include "src/VirtualSensor.h"
#include "src/PointCloud.h"
#include "src/SimpleMesh.h"
bool reconstructRoom(std::string path, std::string outName) {
    std::string filenameIn = path;
    std::string filenameBaseOut = outName;

    // Load video
    std::cout << "Initialize virtual sensor..." << std::endl;
    VirtualSensor sensor;
    if (!sensor.init(filenameIn)) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }
    //Here in Process Next frame Gpu will be used
    sensor.processNextFrame();
    PointCloud target{ sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight() };
    std::cout << sensor.getDepthImageHeight() << std::endl;
    /*std::vector<Matrix4f> estimatedPoses;
    Matrix4f currentCameraToWorld = Matrix4f::Identity();
    estimatedPoses.push_back(currentCameraToWorld.inverse());
    Matrix4f currentCameraPose = currentCameraToWorld.inverse();
    std::cout << "Current camera pose: " << std::endl << currentCameraPose << std::endl;
    estimatedPoses.push_back(currentCameraPose);
    SimpleMesh currentDepthMesh{ sensor, currentCameraPose, 0.1f };
    SimpleMesh currentCameraMesh = SimpleMesh::camera(currentCameraPose, 0.0015f);
    SimpleMesh resultingMesh = SimpleMesh::joinMeshes(currentDepthMesh, currentCameraMesh, Matrix4f::Identity());

    std::stringstream ss;
    ss << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off";
    std::cout << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off" << std::endl;
    if (!resultingMesh.writeMesh(ss.str())) {
        std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
        return -1;
    }*/
    return true;
}
int main()
{

    // In the following cases we should use arrays not vectors
    std::vector<int> A{ 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> B{1, 2, 3, 4, 5, 6, 7, 8};
    int C[8];

    CUDA::my_cuda_func(A, B, C);

    for (auto in : C) {
        std::cout << in << std::endl;
    }
    return reconstructRoom(std::string("../Data/rgbd_dataset_freiburg1_xyz/"), std::string("mesh_"));
    
}
