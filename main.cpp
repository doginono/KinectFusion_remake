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
}