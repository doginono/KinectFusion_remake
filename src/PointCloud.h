#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "SurfaceMeasurement.cuh"
#include <time.h>
#include "Eigen.h"
typedef Eigen::Matrix<unsigned char, 4, 1> Vector4uc;


class PointCloud {
public:

    PointCloud(float* depthMap, const Matrix3f& depthIntrinsics, const Matrix4f& depthExtrinsics, const unsigned width, const unsigned height, unsigned downsampleFactor = 1, float maxDistance = 0.1f) {
        // Get depth intrinsics.
        float fovX = depthIntrinsics(0, 0);
        float fovY = depthIntrinsics(1, 1);
        float cX = depthIntrinsics(0, 2);
        float cY = depthIntrinsics(1, 2);
        const float maxDistanceHalved = maxDistance / 2.f;

        // Compute inverse depth extrinsics.
        Matrix4f depthExtrinsicsInv = depthExtrinsics.inverse();
        Matrix3f rotationInv = depthExtrinsicsInv.block(0, 0, 3, 3);
        Vector3f translationInv = depthExtrinsicsInv.block(0, 3, 3, 1);

        // Back-project the pixel depths into the camera space.
        std::vector<Vector3f> pointsTmp(width * height);
        
        // For every pixel row.
        
        m_depthExtrinsics = depthExtrinsics;
        std::vector<float> camparams = { fovX ,fovY ,cX ,cY };
        CUDA::initSensorFrame(depthMap, rotationInv, translationInv, camparams, pointsTmp);

        std::vector<Vector3f> normalsTmp(width * height);
        CUDA::initnormalMap(depthMap, maxDistanceHalved, normalsTmp);

        //can be parallelized later
        
        const unsigned nPoints = pointsTmp.size();
        m_points.reserve(std::floor(float(nPoints) / downsampleFactor));
        m_normals.reserve(std::floor(float(nPoints) / downsampleFactor));
        
        for (int i = 0; i < nPoints; i = i + downsampleFactor) {
            const auto& point = pointsTmp[i];
            const auto& normal = normalsTmp[i];

            if (point.allFinite() && normal.allFinite()) {
                m_points.push_back(point);
                m_normals.push_back(normal);
            }
        }
        
    }
    
    std::vector<Vector3f>& getPoints() {
        return m_points;
    }

    const std::vector<Vector3f>& getPoints() const {
        return m_points;
    }

    std::vector<Vector3f>& getNormals() {
        return m_normals;
    }

    const std::vector<Vector3f>& getNormals() const {
        return m_normals;
    }
    //CUDA
    unsigned int getClosestPoint(Vector3f& p) {
        unsigned int idx = 0;

        float min_dist = std::numeric_limits<float>::max();
        for (unsigned int i = 0; i < m_points.size(); ++i) {
            float dist = (p - m_points[i]).norm();
            if (min_dist > dist) {
                idx = i;
                min_dist = dist;
            }
        }

        return idx;
    }

public:
    std::vector<Vector3f> m_points;
    std::vector<Vector3f> m_normals;
    Matrix4f m_depthExtrinsics;
};
