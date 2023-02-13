#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include "Eigen.h"
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <algorithm>    // std::min
#include "PointCloud.h"

namespace CUDA {
	//Wrapper to call the kernel function on the GPU
	//an example function about how to call the cuda functions(Wrapper)
	void SurfacePrediction(Vector3d& min, Vector3d& max, double* voxWeights, double* voxValues, Matrix4f& currentCameraPose, Matrix4f& transMatrixcur,
		std::vector<Vector3f>& depthMap, std::vector<Vector3f>& normals, std::vector<float>& camparams, PointCloud& frame);
}