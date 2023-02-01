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
#include "src/SurfaceReconstruction.cuh"
#include <time.h>
#include "src/Volume.h"
#include "src/SurfacePrediction.cuh"

#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 480

/*
#define MinPoint 
#define MaxPoint 
*/

/**
 * A struct encapsulating reoccurring data.
 */
struct Data {
    Matrix4f currentPose{};
    Matrix3f previousRotInv{};
    Matrix4f estimatedPoseBefore{};
    Matrix4f currentCameraPose{};
    float *depthMap;

    Matrix3f depthIntrinsics{};
    std::vector<float> camparams{};
    std::vector<Matrix4f> estimatedPoses{};
};


/*
* First we have two camera space vertices
* the camera spaces are different but still really similar
* Find the correspondences in both
* Update it with the pose estimation
* Assign the value to the next frame depth currentPose needs to be changed
* It will be same with the last frame till you update it
* Use frame to frame tracking
* TODO Extending it with frame to model tracking implement Volume.
*/

/**
 * Generates a synthetic depth map from the fused data in the TSDF.
 */
void generateFrameFromModel(Volume& TSDF, PointCloud sourceFrame, PointCloud frame, Data& _data) {
    // TODO implement depth map generation by raycasting the TSDF
    // frame is what I had the last time will be updated with generate from model
    
    // Idea:shine light from the camera to the volume
    // depthmap from new one not relevant, frame already has its depth map
    std::vector<Vector3f> normalMap(IMAGE_WIDTH * IMAGE_HEIGHT, Vector3f(MINF, MINF, MINF));
    std::vector<Vector3f> points(IMAGE_WIDTH * IMAGE_HEIGHT, Vector3f(MINF, MINF, MINF));
    frame.m_points = points;
    //frame.m_points = normalMap;

    std::cout << _data.currentPose.block<3,3>(0,0) * frame.m_points[19011] << std::endl;
    CUDA::SurfacePrediction(TSDF.min, TSDF.max, TSDF.weights, TSDF.vol, _data.currentCameraPose, _data.currentPose, frame.m_points, frame.m_normals, _data.camparams);
    // After Cuda done delete this add frame= (depth, normals)
    //std::cout << _data.currentPose.block<3, 3>(0, 0)*frame.m_points[305434]+ _data.currentPose.block<3, 1>(0, 3) << std::endl;
    std::cout << frame.m_points[270511] << std::endl;

    // TODO optional: Use FreeImageHelper SaveImageToFile() to visualize the newly created depth map
    //frame = sourceFrame;
}

/**
 * Estimates the transformation and rotation Matrix between two frames.
 * @param iter The current iteration.
 * @param sensor The virtual sensor containing the real-world input data.
 * @param frame The PointCloud containing data of the current frame.
 * @param _data The data wrapper containing matrices etc.
 */
void poseEstimation(const unsigned int iter, VirtualSensor &sensor, PointCloud &frame, Data &_data) {
    //take the transfrom before
    //begin with the previous pose therefore currentPose= estimatedPose[iter] need to update it every iteration in for loop
    //world space
    _data.currentPose = _data.estimatedPoses[iter];
    _data.previousRotInv = _data.estimatedPoses[iter].block<3, 3>(0, 0).inverse(); //To cam coordinates last
    _data.estimatedPoseBefore = _data.estimatedPoses[iter]; // to world coordinates last
    
    _data.depthMap = sensor.getDepth();
    _data.depthIntrinsics = sensor.getDepthIntrinsics();
    float fovX = _data.depthIntrinsics(0, 0);
    float fovY = _data.depthIntrinsics(1, 1);
    float cX = _data.depthIntrinsics(0, 2);
    float cY = _data.depthIntrinsics(1, 2);
    _data.camparams = {fovX, fovY, cX, cY};

    //depthextrinsics always init with identity
    PointCloud sourceFrame{ sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight()};

    //Can be converted to Cuda
    for (int j = 0; j < 2; j++) {
        // First begin with current cameratoworld being identity
        // Update it incrementally for loop change now okay
        //currentPoser updated everytime holds the pose
        //empty vector init with zeros

        // Matrix from camera Space to World space

        std::vector<int> correspondencesArray(IMAGE_WIDTH * IMAGE_HEIGHT, 0); // 640*480 all value zeros
        //estimatedPoses[iter] is the last found pose. which is not updated. hold it for transformations
        CUDA::poseEstimation(sourceFrame, frame, _data.camparams,
                             _data.estimatedPoseBefore, _data.currentPose, _data.previousRotInv, correspondencesArray);
        //400*400 400*400 pixel correspo
        //402*300 400*400


        unsigned nPoints = correspondencesArray.size();

        MatrixXf A = MatrixXf::Zero(nPoints, 6);
        VectorXf b = VectorXf::Zero(nPoints);
        //toworld coordinates current
        Matrix3f rotationtmp = _data.currentPose.block<3, 3>(0, 0);
        Vector3f translationtmp = _data.currentPose.block<3, 1>(0, 3);
        //toworld coordinates last found one
        Matrix3f rotationtmpBefore = _data.estimatedPoseBefore.block<3, 3>(0, 0);
        Vector3f translationtmpBefore = _data.estimatedPoseBefore.block<3, 1>(0, 3);
        //std::cout << "rotationtmp" << rotationtmp << std::endl;
        //std::cout << "rotationtmpBefore " << rotationtmpBefore << std::endl;
        ////std::cout << "translationtmp" << rotationtmp << std::endl;
        ////std::cout << "translationtmpBefore " << translationtmpBefore << std::endl;



        //fillthesystem 640*480
        //make the system smaller
        for (unsigned i = 0; i < nPoints; i++) {
            //worldspace //estimatedPose eklersin
            if (correspondencesArray[i] == 0) {
                continue;
            }
            const Vector3f &s = rotationtmp * sourceFrame.getPoints()[i] + translationtmp; // Frame 2 in global
            const Vector3f &d = rotationtmpBefore * frame.getPoints()[correspondencesArray[i]] + translationtmpBefore;// Frame 1 in global
            const Vector3f &n = rotationtmpBefore * frame.getNormals()[correspondencesArray[i]];


            // TODO: Add the point-to-plane constraints to the system one row
            A(i, 0) = n[2] * s[1] - n[1] * s[2];
            A(i, 1) = n[0] * s[2] - n[2] * s[0];
            A(i, 2) = n[1] * s[0] - n[0] * s[1];
            A(i, 3) = n[0];
            A(i, 4) = n[1];
            A(i, 5) = n[2];
            b(i) = n[0] * d[0] + n[1] * d[1] + n[2] * d[2] - n[0] * s[0] - n[1] * s[1] - n[2] * s[2];
        }

        correspondencesArray.erase(std::remove(begin(correspondencesArray), end(correspondencesArray), 0), end(correspondencesArray));
        nPoints = correspondencesArray.size();
        if (nPoints != 0) {

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
            //currentPose = estimatedPose*currentPose;
            //std::cout << "rotation "    << rotation    << std::endl;
            //std::cout << "translation " << translation << std::endl;
            _data.currentPose.block<3, 3>(0, 0) = rotation * _data.currentPose.block<3, 3>(0, 0);
            _data.currentPose.block<3, 1>(0, 3) = rotation * _data.currentPose.block<3, 1>(0, 3) + translation;
        }
        //std::cout << "how many correspondences: " << nPoints << std::endl;

        ////std::cout << "estimatedPose " << estimatedPose << std::endl;

    }


    //to cam coordinates
    _data.currentCameraPose = _data.currentPose.inverse();
    //currentCameraPose = Matrix4f::Identity();
    //std::cout << "Current camera pose: " << std::endl << currentCameraPose << std::endl;
    //add world coordinate transformation matrix
    _data.estimatedPoses.push_back(_data.currentPose);

}

/**
 * Fuses the data of the frame into the global TSDF model.
 * @param TSDF The global Truncated Signed Distance Function.
 * @param frame The frame to be integrated.
 * @param _data The data wrapper containing matrices etc.
 */
void updateTSDF(Volume &TSDF, PointCloud frame, Data &_data) {
    //Now surface reconstruction
    //voxvalue and weight will be updated everytime
    //add volume h and cpp
    //then init a empty tsdf clean() function
    // then begin initilizing
    CUDA::SurfaceReconstruction(TSDF.min, TSDF.max, TSDF.weights, TSDF.vol, _data.currentCameraPose, _data.currentPose, _data.depthMap, frame.getNormals(), _data.camparams);
    std::cout << *std::max_element(TSDF.vol, TSDF.vol + 512 * 512 * 512) << std::endl;
}

int reconstructRoom(const std::string &path, const std::string &outName) {

    /*
    Vector3f min_point = ;
    Vector3f max_point = ;
    */
    // 512^3 voxels like suggested in the paper, Init the volume with plausible range!!! These are random values which I believe contains all the first frame

    Volume TSDF(Vector3d{-1.5, -1.0, -0.1}, Vector3d{1.5, 1.0, 3.5}, 512, 512, 512, 3);
    //I am oneing out memory if it makes sense. Because if it finds a zero in the ray direction it stops
    TSDF.zeroOutMemory();
    std::cout << TSDF.vol[512] << std::endl;
    Vector3f distanceBetweenVoxels((TSDF.max[0] - TSDF.min[0]) / 511,
                                   (TSDF.max[1] - TSDF.min[1]) / 511,
                                   (TSDF.max[2] - TSDF.min[2]) / 511);
    std::cout << distanceBetweenVoxels << std::endl;

    // Load video
    //std::cout << "Initialize virtual sensor..." << std::endl;
    VirtualSensor sensor;
    if (!sensor.init(path)) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }
    //Here in Process Next frame Gpu will be used
    sensor.processNextFrame();

    PointCloud frame{sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight()};

    Data _data{};
    _data.estimatedPoses.emplace_back(
            Matrix4f::Identity()); //the transformation matrix of the current pose (which will be updated every iteration), normally pass the inverse, but it is identity so don't need it

    // Iterate over frames: do pose estimation (from 3rd frame on calculate previous depth map from model), then integrate into TSDF
    unsigned int iter = 0;
    const unsigned int iMax = 12;

    while (sensor.processNextFrame() && iter <= iMax) {

        // Frame 0:     no pose estimation necessary since it defines the base world pose
        // Frame 1:     frame-to-frame pose estimation (no information gain from TSDF, only contains Frame 0)
        // Frames > 1:  frame-to-model pose estimation (generate artificial frame from TSDF containing fused data)
        
        //came now
        PointCloud sourceFrame{ sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight()};

        if (iter > 1) generateFrameFromModel(TSDF, sourceFrame, frame, _data);
        poseEstimation(iter, sensor, frame, _data);
        updateTSDF(TSDF, sourceFrame, _data);
        //Matrix4f cameraToWorld = _data.currentCameraPose.inverse();
        SimpleMesh currentDepthMesh{ sensor, _data.currentCameraPose, 0.1f };
        SimpleMesh currentCameraMesh = SimpleMesh::camera(_data.currentCameraPose, 0.0015f);
        SimpleMesh resultingMesh = SimpleMesh::joinMeshes(currentDepthMesh, currentCameraMesh, Matrix4f::Identity());
        
        //will be left out after generate

        if (iter % 1 == 0) {
            std::stringstream ss;
            ss << outName << sensor.getCurrentFrameCnt() << ".off";
            std::cout << outName << sensor.getCurrentFrameCnt() << ".off" << std::endl;
            if (!resultingMesh.writeMesh(ss.str())) {
                //std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
                return -1;
            }
        }
        iter++;
    }

    return 0;
}

int main() {
    // In the following cases we should use arrays not vectors
    clock_t t = clock();
    int reconstruction = reconstructRoom(std::string("../Data/rgbd_dataset_freiburg1_xyz/"), std::string("mesh_"));
    t = clock() - t;
    std::cout << "time    " << (float) t / CLOCKS_PER_SEC;
    return reconstruction;
}