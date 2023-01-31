#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "SurfaceReconstruction.cuh"

//voxweights vox values, depthmap, camparams needs to be copied needs to be included
__global__ void surfaceReconstructionKernel(Vector3d min, Vector3d max, double* voxWeights, double* voxValues, Matrix4f currentCameraPose, Matrix4f transMatrixcur,
	float* depthMap, Vector3f* normals, float* camparams)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	// (max[0] - min[0]) / (dx - 1)
	//totaldistance with respect to x y z
	//this can be included in Cuda call
	

}
namespace CUDA {
	//Also need the spacing to be able to project the voxels to ->World->cam->image plane, will be calculated on cuda but needs to be adressed for faster update
	//added min and max point of the voxel, min left lower corner, max= left uppercorner
	void SurfacePrediction(){
		

	}
}
