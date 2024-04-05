#pragma once

#include <iostream>
#include <Eigen/Dense>
//#include<unsupported/Eigen/MatrixFunctions>
#include "chprinter.h"

#include "../source/utils.h"
//====================================================================================
inline void make_data(Eigen::MatrixXd& out_raw,
                      Eigen::MatrixXd& out_times,
                      Eigen::MatrixXd& out_noised_process,
                      Eigen::MatrixXd& out_noised_meas,
                      Eigen::MatrixXd in_x,
                      Eigen::MatrixXd in_model,
                      Eigen::MatrixXd in_G,
                      Eigen::MatrixXd in_Q,
                      Eigen::MatrixXd in_R,
                      Eigen::MatrixXd in_H,//!!!
                      double dt,
                      int iterations,
                      double noise_start=0.,
                      double noise_end=0.)
{
    // == make times =================================
    double time = 0.;
    out_times.resize(1,iterations);
    for(int i=0;i<iterations;i++)
    {
        out_times(0,i) = time;
        time+=dt;
    }
    // ===============================================

    // == make raw =======================================
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
        n << Utils::rnd(noise_start,noise_end), Utils::rnd(noise_start,noise_end), Utils::rnd(noise_start,noise_end), Utils::rnd(noise_start,noise_end), Utils::rnd(noise_start,noise_end), Utils::rnd(noise_start,noise_end);
        noise.col(i) = n;
    }
    out_noised_process = out_raw + Utils::sqrt_one_by_one(in_G*in_Q*in_G.transpose())*noise;
    // ===================================================

    // == noised - 2 =====================================
    Eigen::MatrixXd n2((in_H*out_noised_process).rows(),x.cols());
    Eigen::MatrixXd noise2((in_H*out_noised_process).rows(),iterations);
    for(int i=0;i<iterations;i++)
    {
        n2 << Utils::rnd(noise_start,noise_end), Utils::rnd(noise_start,noise_end), Utils::rnd(noise_start,noise_end);
        noise2.col(i) = n2;
    }
    out_noised_meas = in_H*out_noised_process + Utils::sqrt_one_by_one(in_R)*noise2;
    // ===================================================

}
//====================================================================================
