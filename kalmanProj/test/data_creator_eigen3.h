#pragma once

#include <iostream>
#include <eigen3/Eigen/Dense>
#include<unsupported/Eigen/MatrixFunctions>
#include "printer_eigen3.h"
#include <chrono>
#include <random>
#include "Eigen/Dense"
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
Eigen::MatrixXd sqrt6x6(Eigen::MatrixXd A)
{
    Eigen::MatrixXd S(A.rows(),A.cols());
    double A00=A(0,0),A01=A(0,1),A02=A(0,2),A03=A(0,3),A04=A(0,4),A05=A(0,5);
    double            A11=A(1,1),A12=A(1,2),A13=A(1,3),A14=A(1,4),A15=A(1,5);
    double                       A22=A(2,2),A23=A(2,3),A24=A(2,4),A25=A(2,5);
    double                                  A33=A(3,3),A34=A(3,4),A35=A(3,5);
    double                                             A44=A(4,4),A45=A(4,5);
    double                                                        A55=A(5,5);

    double S00=sqrt(A00);
    double S01=A01/S00;
    double S02=A02/S00;
    double S03=A03/S00;
    double S04=A04/S00;
    double S05=A05/S00;

    double S11=sqrt(A11-S01*S01);
    double S12=(A12-S01*S02)/S11;
    double S13=(A13-S01*S03)/S11;
    double S14=(A14-S01*S04)/S11;
    double S15=(A15-S01*S05)/S11;

    double S22=sqrt(A22-S02*S02-S12*S12);
    double S23=(A23-S02*S03-S12*S13)/S22;
    double S24=(A24-S02*S04-S12*S14)/S22;
    double S25=(A25-S02*S05-S12*S15)/S22;

    double S33=sqrt(A33-S03*S03-S13*S13-S23*S23);
    double S34=(A34-S03*S04-S13*S14-S23*S24)/S33;
    double S35=(A35-S03*S05-S13*S15-S23*S25)/S33;

    double S44=sqrt(A44-S04*S04-S14*S14-S24*S24-S34*S34);
    double S45=(A45-S04*S05-S14*S15-S24*S25-S34*S35)/S44;

    double S55=sqrt(A55-S05*S05-S15*S15-S25*S25-S35*S35-S45*S45);


    S << S00,S01,S02,S03,S04,S05,
          0.,S11,S12,S13,S14,S15,
          0., 0.,S22,S23,S24,S25,
          0., 0., 0.,S33,S34,S35,
          0., 0., 0., 0.,S44,S45,
          0., 0., 0., 0., 0.,S55;

    return S;
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
std::pair<Eigen::MatrixXd,Eigen::MatrixXd> make_data(Eigen::MatrixXd in_x, Eigen::MatrixXd in_model, Eigen::MatrixXd in_GQG, int iterations, double noise_start=0., double noise_end=0.)
{
    Eigen::MatrixXd out_noised(in_x.cols(),iterations);
    Eigen::MatrixXd out_clear(in_x.cols(),iterations);
    Eigen::MatrixXd x = in_x.transpose();

    for(int i=0;i<iterations;i++)
    {
        out_clear.col(i) = x;
        x = in_model*x;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> aa(in_GQG);
    Eigen::MatrixXd sqrtGQG = aa.operatorSqrt();

    //=============================================
    sqrtGQG << 0.02, 0.063, 0., 0., 0., 0.,
               0.063, 0.2, 0., 0., 0., 0.,
               0., 0., 0.02, 0.063, 0., 0.,
               0., 0., 0.063, 0.2, 0., 0.,
               0., 0., 0., 0., 0.02, 0.063,
               0., 0., 0., 0., 0.063, 0.2;//#TEMP
    //=============================================

    Eigen::MatrixXd n(x.rows(),x.cols());
    Eigen::MatrixXd noise(x.rows(),iterations);
    for(int i=0;i<iterations;i++)
    {
        n << rnd(noise_start,noise_end), rnd(noise_start,noise_end), rnd(noise_start,noise_end), rnd(noise_start,noise_end), rnd(noise_start,noise_end), rnd(noise_start,noise_end);
        noise.col(i) = n;
    }
    out_noised = out_clear + sqrtGQG*noise;

    return std::make_pair(out_clear,out_noised);
}
//====================================================================================
std::pair<Eigen::MatrixXd,Eigen::MatrixXd> make_data2(Eigen::MatrixXd in_x,
                                                      Eigen::MatrixXd in_model,
                                                      Eigen::MatrixXd in_G,
                                                      Eigen::MatrixXd in_Q,
                                                      Eigen::MatrixXd in_R,
                                                      int iterations,
                                                      double noise_start=0.,
                                                      double noise_end=0.)
{
    // == make raw =====================================
    Eigen::MatrixXd out_clear(in_x.cols(),iterations);
    Eigen::MatrixXd x = in_x.transpose();
    for(int i=0;i<iterations;i++)
    {
        out_clear.col(i) = x;
        x = in_model*x;
    }
    // ===================================================

    // == matrix =========================================
    Eigen::MatrixXd GQG = in_G*in_Q*in_G.transpose();

    // == sqrt-1 ==
    Eigen::LLT<Eigen::MatrixXd> lltofP(GQG);
    if(lltofP.info() != Eigen::Success)
        std::cout << "sqrtGQG1 - BAD_SQRT!" << std::endl;
    Eigen::MatrixXd sqrtGQG1 = lltofP.matrixL();

    Eigen::LLT<Eigen::MatrixXd> lltofP0(in_R);
    if(lltofP.info() != Eigen::Success)
        std::cout << "sqrtR1 - BAD_SQRT!" << std::endl;
    Eigen::MatrixXd sqrtR1 = lltofP0.matrixL();
    // ============

    // == sqrt-2 ==
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> aa(GQG);
    Eigen::MatrixXd sqrtQ2 = aa.operatorSqrt();
    // ============

    // == sqrt-3 ==
    Eigen::MatrixXd sqrtGQG3 = sqrt6x6(GQG);
    // ============

    // == sqrt-4 ==
    Eigen::MatrixXd sqrtGQG4 = GQG.sqrt();
    // ============

    Eigen::MatrixXd sqrtGQG = sqrtGQG1;
    Eigen::MatrixXd sqrtR = sqrtR1;
    // ===================================================

    // == noised - 1 =====================================
    Eigen::MatrixXd n(x.rows(),x.cols());
    Eigen::MatrixXd noise(x.rows(),iterations);
    for(int i=0;i<iterations;i++)
    {
        n << rnd(noise_start,noise_end), rnd(noise_start,noise_end), rnd(noise_start,noise_end), rnd(noise_start,noise_end), rnd(noise_start,noise_end), rnd(noise_start,noise_end);
        noise.col(i) = n;
    }
    Eigen::MatrixXd out_noised = out_clear + sqrtGQG*noise;
    // ===================================================

    Eigen::MatrixXd H(3,6);
    H << 1., 0., 0., 0., 0., 0.,
         0., 0., 1., 0., 0., 0.,
         0., 0., 0., 0., 1., 0.;

    // == noised - 2 =====================================
    Eigen::MatrixXd n2((H*out_noised).rows(),x.cols());
    Eigen::MatrixXd noise2((H*out_noised).rows(),iterations);
    for(int i=0;i<iterations;i++)
    {
        n2 << rnd(noise_start,noise_end), rnd(noise_start,noise_end), rnd(noise_start,noise_end);
        noise2.col(i) = n2;
    }
    Eigen::MatrixXd out_noised3 = H*out_noised + sqrtR*noise2;
    // ===================================================

    //- charts -------------------------------------
    std::vector<MTN2> mtn_vec;
    Eigen::VectorXd data_clear_x = out_clear.row(0);
    Eigen::VectorXd data_clear_y = out_clear.row(2);
    Eigen::VectorXd data_clear_z = out_clear.row(4);
    Eigen::VectorXd data_noised_x = out_noised.row(0);
    Eigen::VectorXd data_noised_y = out_noised.row(2);
    Eigen::VectorXd data_noised_z = out_noised.row(4);
    Eigen::VectorXd data_noised3_x = out_noised3.row(0);
    Eigen::VectorXd data_noised3_y = out_noised3.row(1);
    Eigen::VectorXd data_noised3_z = out_noised3.row(2);
    mtn_vec.push_back(MTN2(TYPE::LINE,"data test: y(x)",
                           std::vector<double>(data_clear_y.data(),data_clear_y.data()+data_clear_y.size()),
                           std::vector<double>(data_clear_x.data(),data_clear_x.data()+data_clear_x.size())));
    mtn_vec.push_back(MTN2(TYPE::LINE,"data test: y(x)",
                           std::vector<double>(data_noised_y.data(),data_noised_y.data()+data_noised_y.size()),
                           std::vector<double>(data_noised_x.data(),data_noised_x.data()+data_noised_x.size())));
    mtn_vec.push_back(MTN2(TYPE::LINE,"data test: y(x)",
                           std::vector<double>(data_noised3_y.data(),data_noised3_y.data()+data_noised3_y.size()),
                           std::vector<double>(data_noised3_x.data(),data_noised3_x.data()+data_noised3_x.size())));
    print_charts_universal3(mtn_vec);
    //----------------------------------------------

    return std::make_pair(out_clear,out_noised3);
}
//====================================================================================
