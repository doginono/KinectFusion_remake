#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "src/SurfaceMeasurement.h"
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/NonLinearOptimization>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

int main()
{
    Eigen::Vector3f x(1, 2, 3);
    std::cout << x << std::endl;
    //int* elements = new int[9] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    int A[9]{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    int B[9]{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    int C[9]{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    
    //my_cuda_func(A, B, C);
    for (auto in : C) {
        std::cout << in<< std::endl;
    }

    return 0;
}
