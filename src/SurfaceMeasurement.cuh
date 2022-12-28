#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include "Eigen.h"

namespace CUDA{
	//Wrapper to call the kernel function on the GPU
	//an example function about how to call the cuda functions(Wrapper)
	void example(float depthMap[],std::vector<Vector3f>& pointsTmp);
	void initSensorFrame(float depthMap[], Matrix3f& rotationInv,  Vector3f& translationInv,
		std::vector<float>& camparams, std::vector<Vector3f>& pointsTmp) ;
}