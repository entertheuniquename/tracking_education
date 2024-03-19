#pragma once

#include <iostream>
#include <eigen3/Eigen/Dense>
#include<unsupported/Eigen/MatrixFunctions>
#include "chprinter.h"
#include <chrono>
#include <random>
#include "Eigen/Dense"
//====================================================================================
Eigen::MatrixXd sqrt_one_by_one(Eigen::MatrixXd A)
{
    Eigen::MatrixXd B(A.cols(),A.rows());
    for(int i=0;i<A.cols();i++)
        for(int j=0;j<A.rows();j++)
            B(i,j) = std::sqrt(A(i,j));
    return B;
}
//====================================================================================
int eigen3_matrix_check(Eigen::MatrixXd A)
{   //--------------------------------------------------
    Eigen::LLT<Eigen::MatrixXd> lltofCovariance(A);;
    if(lltofCovariance.info() == Eigen::NumericalIssue);
        return 1;
    //--------------------------------------------------
    if(!A.transpose().isApprox(A,0.00000001))
        return 2;
    //--------------------------------------------------
    if(A.determinant() == 0)
        return 3;
    //--------------------------------------------------
    return 0;
}
//====================================================================================
//random generator
double rnd(double s, double e)
{
    using namespace std::chrono;
    int is = s*1000, ie = e*1000;

    std::mt19937 gen;
    gen.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist;

    unsigned long long t = dist(gen);
    srand(t);

    return (rand() % (ie - is + 1) +is)/1000.;
}
//====================================================================================
void make_data(Eigen::MatrixXd& out_raw,
               Eigen::MatrixXd& out_noised_process,
               Eigen::MatrixXd& out_noised_meas,
               Eigen::MatrixXd in_x,
               Eigen::MatrixXd in_model,
               Eigen::MatrixXd in_G,
               Eigen::MatrixXd in_Q,
               Eigen::MatrixXd in_R,
               Eigen::MatrixXd in_H,//!!!
               int iterations,
               double noise_start=0.,
               double noise_end=0.)
{
    // == make raw =====================================
    out_raw.resize(in_x.cols(),iterations);
    Eigen::MatrixXd x = in_x.transpose();
    for(int i=0;i<iterations;i++)
    {
        out_raw.col(i) = x;
        x = in_model*x;
    }
    // ===================================================

    // == noised - 1 =====================================
    Eigen::MatrixXd n(x.rows(),x.cols());
    Eigen::MatrixXd noise(x.rows(),iterations);
    for(int i=0;i<iterations;i++)
    {
        n << rnd(noise_start,noise_end), rnd(noise_start,noise_end), rnd(noise_start,noise_end), rnd(noise_start,noise_end), rnd(noise_start,noise_end), rnd(noise_start,noise_end);
        noise.col(i) = n;
    }
    out_noised_process = out_raw + sqrt_one_by_one(in_G*in_Q*in_G.transpose())*noise;
    // ===================================================

    // == noised - 2 =====================================
    Eigen::MatrixXd n2((in_H*out_noised_process).rows(),x.cols());
    Eigen::MatrixXd noise2((in_H*out_noised_process).rows(),iterations);
    for(int i=0;i<iterations;i++)
    {
        n2 << rnd(noise_start,noise_end), rnd(noise_start,noise_end), rnd(noise_start,noise_end);
        noise2.col(i) = n2;
    }
    out_noised_meas = in_H*out_noised_process + sqrt_one_by_one(in_R)*noise2;
    // ===================================================

}
//====================================================================================
