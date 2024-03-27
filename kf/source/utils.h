#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <random>

namespace Utils
{
Eigen::MatrixXd transpose(Eigen::MatrixXd in)
{
    return in.transpose();
}

Eigen::MatrixXd inverse(Eigen::MatrixXd in)
{
    return in.inverse();
}
void progress_print(int am, int i, int per_step=1,std::string s="")
{
    static int c=0;
    static bool bs=false;
    if(!bs){std::cout << s;bs=true;}
    if(i/(am/100.)>c){std::cout << "["+std::to_string(c)+"%]" << std::flush;c+=per_step;};
    if(i==am-1){std::cout << "["+std::to_string(c)+"%]" << std::endl;c=0;bs=false;}
}
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
Eigen::MatrixXd mpow_obo(Eigen::MatrixXd x)
{
    return (x.array()*x.array()).matrix();
}
std::vector<double> mrow(Eigen::MatrixXd m,int n)
{
    Eigen::VectorXd v = m.row(n);
    return  std::vector<double>(v.data(),v.data()+v.size());
}
std::vector<double> mcol(Eigen::MatrixXd m,int n)
{
    Eigen::VectorXd v = m.col(n);
    return  std::vector<double>(v.data(),v.data()+v.size());
}
Eigen::MatrixXd sqrt_one_by_one(Eigen::MatrixXd A)
{
    Eigen::MatrixXd B(A.rows(),A.cols());
    for(int i=0;i<A.cols();i++)
        for(int j=0;j<A.rows();j++)
            B(j,i) = std::sqrt(A(j,i));
    return B;
}
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
}
