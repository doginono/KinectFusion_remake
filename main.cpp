#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "src/SurfaceMeasurement.cuh"
#include "src/PoseEstimation.cuh"
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
   
    std::vector<Matrix4f> estimatedPoses;

    //the transformation matrix of the current pose (which will be updated every iteration)
    Matrix4f transMatrixcur = Matrix4f::Identity();
    
    //normally pass the inverse but it is identity so dont need it
    estimatedPoses.push_back(transMatrixcur);

    //PointCloud previous = target;
    std::vector<int> tempo;
    //init emtyp vector
    for (int i = 0; i < 640 * 480; i++) {
        tempo.push_back(0);
    }
    int iter = 0;
    const int iMax = 2;
    /* 
    * First we have two camera space vertices
    * the camera spaces are different but still really similar
    * Find the correspondences in both
    * Update it with the pose estimation
    * Assign the value to the next frame depth transMatrixcur needs to be changed
    * It will be same with the last frame till you update it
    * Use frame to frame tracking
    */
    while (sensor.processNextFrame() && iter <= iMax) {

        //take the transfrom before
        //begin with the previous pose therefore transMatrixcur= estimatedPose[iter] need to update it every iteration in for loop
        //world space
        transMatrixcur = estimatedPoses[iter];

		float* depthMap = sensor.getDepth();
		Matrix3f depthIntrinsics = sensor.getDepthIntrinsics();
        //depthextrinsics always init with identity
		PointCloud source{ depthMap, sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight()};

        //find the pose regarding the last frame
        float fovX = depthIntrinsics(0, 0);
        float fovY = depthIntrinsics(1, 1);
        float cX = depthIntrinsics(0, 2);
        float cY = depthIntrinsics(1, 2);
        std::vector<float> camparams = { fovX ,fovY ,cX ,cY };

        for(int j=0;j<5;j++){
        // First begin with current cameratoworld being identity
        // Update it incrementally for loop change now okay
            //transmatrixcurr updated everytime holds the pose
            Matrix4f worldToCam = transMatrixcur.inverse();

            //empty vector init with zeros
            std::vector<int> correspondencesArray = tempo;
            //world tocam not needed?
            //estimatedPoses[iter] is the last found pose. which is not updated. hold it for transformations
            CUDA::poseEstimation(source, target, camparams, 
                estimatedPoses[iter], transMatrixcur, worldToCam, correspondencesArray);

            unsigned nPoints = correspondencesArray.size();
            correspondencesArray.erase(std::remove(begin(correspondencesArray), end(correspondencesArray), 0), end(correspondencesArray));


            nPoints = correspondencesArray.size();
            std::cout << "how many correspondences: " << nPoints << std::endl;
            std::cout << "image number of pixels: " << 480 * 640 << std::endl;
            
            MatrixXf A = MatrixXf::Zero(4 * nPoints, 6);
            VectorXf b = VectorXf::Zero(4 * nPoints);
            Matrix3f rotationtmp = transMatrixcur.block<3, 3>(0, 0);
            Vector3f translationtmp = transMatrixcur.block<3, 1>(0, 3);
            Matrix4f toGlobal = estimatedPoses[iter].inverse();
            std::cout << "toGlobal " << toGlobal << std::endl;
            std::cout << "transMatrixcur " << transMatrixcur << std::endl;

            //fillthesystem
            for (unsigned i = 0; i < nPoints; i++)
            {
                //worldspace
                const auto& s = rotationtmp * source.getPoints()[correspondencesArray[i]] + translationtmp;
                const auto& d = toGlobal.block<3, 3>(0, 0) * target.getPoints()[correspondencesArray[i]] + toGlobal.block<3, 1>(0, 3);
                const auto& n = toGlobal.block<3, 3>(0, 0) * target.getNormals()[correspondencesArray[i]];

                // TODO: Add the point-to-plane constraints to the system one row
                A(4 * i, 0) = n[2] * s[1] - n[1] * s[2];
                A(4 * i, 1) = n[0] * s[2] - n[2] * s[0];
                A(4 * i, 2) = n[1] * s[0] - n[0] * s[1];
                A(4 * i, 3) = n[0];
                A(4 * i, 4) = n[1];
                A(4 * i, 5) = n[2];
                b(4 * i) = n[0] * d[0] + n[1] * d[1] + n[2] * d[2] - n[0] * s[0] - n[1] * s[1] - n[2] * s[2];

                A.row(4 * i + 1) << 0, s[2], -s[1], 1, 0, 0;
                A.row(4 * i + 2) << -s[2], 0, s[0], 0, 1, 0;
                A.row(4 * i + 3) << s[1], -s[0], 0, 0, 0, 1;
                b(4 * i + 1) = d[0] - s[0];
                b(4 * i + 2) = d[1] - s[1];
                b(4 * i + 3) = d[2] - s[2];

            }

            // TODO: Solve the system ans!!!
            VectorXf x(6);
            x = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b); 
            float alpha = x(0), beta = x(1), gamma = x(2);

            // Build the pose matrix
            Matrix3f rotation = AngleAxisf(alpha, Vector3f::UnitX()).toRotationMatrix() *
                AngleAxisf(beta, Vector3f::UnitY()).toRotationMatrix() *
                AngleAxisf(gamma, Vector3f::UnitZ()).toRotationMatrix();

            Vector3f translation = x.tail(3);

            // TODO: Build the pose matrix using the rotation and translation matrices
            //transMatrixcur = estimatedPose*transMatrixcur;
            std::cout << "rotation " << rotation << std::endl;
            std::cout << "translation " << translation << std::endl;

            transMatrixcur.block<3, 3>(0, 0) = rotation * transMatrixcur.block<3, 3>(0, 0);
            transMatrixcur.block<3, 1>(0, 3) = rotation * transMatrixcur.block<3, 1>(0, 3) + translation;

            //std::cout << "estimatedPose " << estimatedPose << std::endl;
            std::cout << "transMatrixcur " << transMatrixcur << std::endl;
        }
        // transMatrixcur *= 2;
		// Invert the transformation matrix to get the current camera pose.
        Matrix4f currentCameraPose = transMatrixcur;// * estimatedPoses[iter]);
        
		std::cout << "Current camera pose: " << std::endl << currentCameraPose << std::endl;
		estimatedPoses.push_back(currentCameraPose);

        SimpleMesh currentDepthMesh{ sensor, currentCameraPose, 0.1f };
        SimpleMesh currentCameraMesh = SimpleMesh::camera(currentCameraPose, 0.0015f);
        SimpleMesh resultingMesh = SimpleMesh::joinMeshes(currentDepthMesh, currentCameraMesh, Matrix4f::Identity());

        if (iter % 1 == 0) {
            std::stringstream ss;
            ss << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off";
            std::cout << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off" << std::endl;
            if (!resultingMesh.writeMesh(ss.str())) {
                std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
                return -1;
            }
        }
        target = source;
		iter++;
	}
    return true;
}
int main()
{

    // In the following cases we should use arrays not vectors
    bool yokArtik = reconstructRoom(std::string("../Data/rgbd_dataset_freiburg1_xyz/"), std::string("mesh_"));
    return yokArtik;
    
}
