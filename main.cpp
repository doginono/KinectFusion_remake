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
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */

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
 
    return true;
}
int main()
{

    // In the following cases we should use arrays not vectors

    return reconstructRoom(std::string("../Data/rgbd_dataset_freiburg1_xyz/"), std::string("mesh_"));
    
}
