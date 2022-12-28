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
        //paralelize with GPUUUU
        // 
        

        CUDA::example(depthMap,pointsTmp);

        std::vector<float> camparams = { fovX ,fovY ,cX ,cY };
        
        CUDA::initSensorFrame(depthMap, rotationInv, translationInv, camparams, pointsTmp);
/*
#pragma omp parallel for
        for (int v = 0; v < height; ++v) {
            // For every pixel in a row.
            for (int u = 0; u < width; ++u) {
                unsigned int idx = v * width + u; // linearized index
                float depth = depthMap[idx];
                if (depth == MINF) {
                    pointsTmp[idx] = Vector3f(MINF, MINF, MINF);
                }
                else {
                    // this equation is same as the equation (3) on the kinectfusion paper
                    // Calculate it to get the position of the points from the sensors point of view
                    // After calculating it will be possible to optimize as the difference between frames will be minimal => ICP
                    // Back-projection to camera space.
                    pointsTmp[idx] = rotationInv * Vector3f((u - cX) / fovX * depth, (v - cY) / fovY * depth, depth) + translationInv;
                }
            }
        }

*/
        //
        // We need to compute derivatives and then the normalized normal vector (for valid pixels).
        std::vector<Vector3f> normalsTmp(width * height);
//Parallelize for normals CUDA
#pragma omp parallel for
        for (int v = 1; v < height - 1; ++v) {
            for (int u = 1; u < width - 1; ++u) {
                unsigned int idx = v * width + u; // linearized index

                const float du = 0.5f * (depthMap[idx + 1] - depthMap[idx - 1]);
                const float dv = 0.5f * (depthMap[idx + width] - depthMap[idx - width]);
                if (!std::isfinite(du) || !std::isfinite(dv) || abs(du) > maxDistanceHalved || abs(dv) > maxDistanceHalved) {
                    normalsTmp[idx] = Vector3f(MINF, MINF, MINF);
                    continue;
                }

                // TODO: Compute the normals using central differences. 
                //normalsTmp[idx] = Vector3f(1, 1, 1); // Needs to be replaced.
                normalsTmp[idx] = Vector3f(du, -dv, 1);
                normalsTmp[idx].normalize();
            }
        }

        // We set invalid normals for border regions.
        for (int u = 0; u < width; ++u) {
            normalsTmp[u] = Vector3f(MINF, MINF, MINF);
            normalsTmp[u + (height - 1) * width] = Vector3f(MINF, MINF, MINF);
        }
        for (int v = 0; v < height; ++v) {
            normalsTmp[v * width] = Vector3f(MINF, MINF, MINF);
            normalsTmp[(width - 1) + v * width] = Vector3f(MINF, MINF, MINF);
        }

        // We filter out measurements where either point or normal is invalid.
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

private:
    std::vector<Vector3f> m_points;
    std::vector<Vector3f> m_normals;

};
