#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>

namespace CUDA{
	//Wrapper to call the kernel function on the GPU
	void my_cuda_func(std::vector<int>& A, std::vector<int>& B, int C[]);
}