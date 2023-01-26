#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "SurfaceMeasurement.cuh"
#include <stdio.h>
#include <iostream>
#include <time.h>
#include "PoseEstimation.cuh"

//The GPU does the work
//add also normal constraints
__global__ void poseEstimation_kernel(float *camparams, Vector3f *verticesSource, Vector3f *verticesPrevious, Vector3f *normalsSource, Vector3f *normalsPrevious,
                                      Matrix4f estimatedPose, Matrix4f transMatrixcur, Matrix3f previousRotInv, int *correspondencesArray) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < 640 * 480) { //640*480 being the height and width //if correspondence found write the index of the found to the correspondencesarray
        if (verticesSource[tid] != Vector3f(MINF, MINF, MINF) && verticesPrevious[tid] != Vector3f(MINF, MINF, MINF) &&
            verticesPrevious[tid][2] != MINF && verticesSource[tid][2] != MINF && normalsPrevious[tid] != Vector3f(MINF, MINF, MINF)) {
            //at first init with Ti=Ti-1 we are first in camspace
            // TransMatrixcurr is brings to cam coordinates maybe wrong
            Vector3f currentVertexGlobal = transMatrixcur.block<3, 3>(0, 0) * verticesSource[tid] + transMatrixcur.block<3, 1>(0, 3);
            // now transform this global vertex to the cameraspace of the frame before
            // Same as the paper (perspective projection)
            // Only difference is that I perspective projected the vertex from the last frame not frame before
            //we store in cameraspace estimatedPose no need to invert

//Vector3f currentVertexCamSpace = previousRotInv * (currentVertexGlobal - estimatedPose.block<3, 1>(0, 3));
            Vector3f currentVertexCamSpace = previousRotInv * (currentVertexGlobal - estimatedPose.block<3, 1>(0, 3));

//Now the perspective projection to pixel coordinates to find the corresponding Vertex
            //Now we have the pixel coordinates for the frame before
            Vector2i pixelCoord(int(currentVertexCamSpace[0] * camparams[0] / currentVertexCamSpace[2] + camparams[2]),
                                int(currentVertexCamSpace[1] * camparams[1] / currentVertexCamSpace[2] + camparams[3]));

            if (pixelCoord[0] > 0 && pixelCoord[0] < 640 && pixelCoord[1] > 0 && pixelCoord[1] < 480) {
                //to world coordinates estimatedPose transforms in world coordinates
                Vector3f vertexPrevGlobal = estimatedPose.block<3, 3>(0, 0) * verticesPrevious[pixelCoord[1] * 640 + pixelCoord[0]] + estimatedPose.block<3, 1>(0, 3);

                //here the normals are also considered that they are not much different meaning that the vertices can be matched
                if ((vertexPrevGlobal - currentVertexGlobal).norm() < 1 && normalsSource[tid].norm() != MINF &&
                    normalsPrevious[pixelCoord[1] * 640 + pixelCoord[0]].norm() != MINF &&
                    (transMatrixcur.block<3, 3>(0, 0) * normalsSource[tid]).cross(estimatedPose.block<3, 3>(0, 0) * normalsPrevious[pixelCoord[1] * 640 + pixelCoord[0]]).norm() > 0.2) {
                    //printf("Difference, Coordinates %f %i %i \n", (vertexPrevGlobal - currentVertexGlobal).norm(), pixelCoord[0] , pixelCoord[1]);
                    correspondencesArray[tid] = pixelCoord[1] * 640 + pixelCoord[0];
                }
            }
        }
    }
}


namespace CUDA {
    //Wrapper to call the kernel function on the GPU
    //only going to update the variable camToWorld to find the transformation

    void poseEstimation(PointCloud &source, PointCloud &previous, std::vector<float> &camparams,
                        Matrix4f &lastpose, Matrix4f &transMatrixcur, Matrix3f &previousRotInv, std::vector<int> &correspondencesArray) {
        // currentcamerapose.inverse()*target

        Vector3f *verticesSource;
        Vector3f *verticesPrevious;

        Vector3f *normalsSource;
        Vector3f *normalsPrevious;

        float *camparamPointer; //params of the source
        int *correspondencesPointer;

        //Mallocs
        //4 variables in camparams Look at exercise 5 for multiplication
        cudaMalloc(&camparamPointer, sizeof(float) * 4);
        cudaMalloc((void **) &verticesSource, sizeof(Vector3f) * 640 * 480);
        cudaMalloc((void **) &verticesPrevious, sizeof(Vector3f) * 640 * 480);
        cudaMalloc((void **) &normalsSource, sizeof(Vector3f) * 640 * 480);
        cudaMalloc((void **) &normalsPrevious, sizeof(Vector3f) * 640 * 480);
        cudaMalloc(&correspondencesPointer, sizeof(int) * 640 * 480);


        cudaMemcpy(verticesSource, source.getPoints().data(), sizeof(Vector3f) * 640 * 480, cudaMemcpyHostToDevice);
        cudaMemcpy(verticesPrevious, previous.getPoints().data(), sizeof(Vector3f) * 640 * 480, cudaMemcpyHostToDevice);
        cudaMemcpy(normalsSource, source.getNormals().data(), sizeof(Vector3f) * 640 * 480, cudaMemcpyHostToDevice);
        cudaMemcpy(normalsPrevious, previous.getNormals().data(), sizeof(Vector3f) * 640 * 480, cudaMemcpyHostToDevice);
        cudaMemcpy(correspondencesPointer, correspondencesArray.data(), sizeof(int) * 640 * 480, cudaMemcpyHostToDevice);
        cudaMemcpy(camparamPointer, camparams.data(), sizeof(float) * 4, cudaMemcpyHostToDevice);
        //later you change
        poseEstimation_kernel <<<307200, 1 >>>(camparamPointer, verticesSource, verticesPrevious, normalsSource, normalsPrevious, lastpose,
                                               transMatrixcur, previousRotInv, correspondencesPointer);

        cudaMemcpy(correspondencesArray.data(), correspondencesPointer, sizeof(int) * 640 * 480, cudaMemcpyDeviceToHost);

        cudaFree(verticesSource);
        cudaFree(verticesPrevious);
        cudaFree(normalsSource);
        cudaFree(normalsPrevious);
        cudaFree(camparamPointer);
        cudaFree(correspondencesPointer);
        //delete[] correspondencesArray;
    }

}