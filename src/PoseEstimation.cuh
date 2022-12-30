#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include "PointCloud.h"
#include "Eigen.h"

namespace CUDA {
	//Wrapper to call the kernel function on the GPU
	//an example function about how to call the cuda functions(Wrapper)
	void poseEstimation(PointCloud& source, PointCloud& previous, std::vector<float>& camparams,
		Matrix4f& currentCameraToWorld, Matrix4f& worldToCamera, std::vector<int>& correspondencesArray);
}