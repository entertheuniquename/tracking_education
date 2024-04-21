#pragma once

#include <Eigen/Dense>
#include <Eigen/QR>
#include <iostream>
#include <chrono>
#include <random>
#include <armadillo>
#include <complex>
#include <algorithm>

using namespace Eigen;

namespace Utils
{
inline Eigen::MatrixXd transpose(Eigen::MatrixXd in)
{
    return in.transpose();
}

inline Eigen::MatrixXd inverse(Eigen::MatrixXd in)
{
    return in.inverse();
}

inline Eigen::MatrixXd diag(std::vector<double> v)
{
    Eigen::MatrixXd M(v.size(),v.size());
    M.setZero();
    for(unsigned long i=0;i<v.size();i++)
        M(i,i) = v[i];
    return M;
}

inline Eigen::MatrixXd state_vector(std::vector<double> v)
{
    Eigen::MatrixXd M(1,v.size());
    M.setZero();
    for(unsigned long i=0;i<v.size();i++)
        M(i,0) = v[i];
    return M;
}

inline void progress_print(int am, int i, int per_step=1,std::string s="")
{
    static int c=0;
    static bool bs=false;
    if(!bs){std::cout << s;bs=true;}
    if(i/(am/100.)>c){std::cout << "["+std::to_string(c)+"%]" << std::flush;c+=per_step;};
    if(i==am-1){std::cout << "["+std::to_string(c)+"%]" << std::endl;c=0;bs=false;}
}
//random generator
inline double rnd(double s, double e)
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
inline Eigen::MatrixXd mpow_obo(Eigen::MatrixXd x)
{
    return (x.array()*x.array()).matrix();
}
inline std::vector<double> mrow(Eigen::MatrixXd m,int n)
{
    Eigen::VectorXd v = m.row(n);
    return  std::vector<double>(v.data(),v.data()+v.size());
}
inline std::vector<double> mcol(Eigen::MatrixXd m,int n)
{
    Eigen::VectorXd v = m.col(n);
    return  std::vector<double>(v.data(),v.data()+v.size());
}
inline Eigen::MatrixXd sqrt_one_by_one(Eigen::MatrixXd A)
{
    Eigen::MatrixXd B(A.rows(),A.cols());
    for(int i=0;i<A.cols();i++)
        for(int j=0;j<A.rows();j++)
            B(j,i) = std::sqrt(A(j,i));
    return B;
}
inline int eigen3_matrix_check(Eigen::MatrixXd A)
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

inline double eps(const double& x = 1.0)
{
    auto xp = std::abs(x);
    return std::nextafter(xp, xp + 1.0f) - xp;
}

inline size_t length(const Eigen::MatrixXd& m)
{
    return m.rows() * m.cols();
}

inline Eigen::MatrixXd zeros(size_t n, size_t m)
{
    Eigen::MatrixXd M(n,m);
    M.setZero();
    return M;
}

inline Eigen::MatrixXd ComputeKalmanGain_E(Eigen::MatrixXd Sy,Eigen::MatrixXd Pxy)
{
    Eigen::MatrixXd B = Sy.triangularView<Eigen::Lower>();
    Eigen::MatrixXd K1 = B.ldlt().solve(Pxy);
    Eigen::MatrixXd K2 = B.ldlt().solve(K1);
    Eigen::MatrixXd K  = K2.transpose();
    return K;
}

inline Eigen::MatrixXd AE(arma::Mat<double> a)
{
    return Eigen::Map<Eigen::MatrixXd>(a.memptr(),a.n_rows,a.n_cols);
}

inline arma::Mat<double> EA(Eigen::MatrixXd e)
{
    return arma::Mat<double>(e.data(),e.rows(), e.cols(),true,false);
}

//#TODO
inline Eigen::MatrixXd qrFactor(const Eigen::MatrixXd& A,
                                const Eigen::MatrixXd& S,
                                const Eigen::MatrixXd& Ns)
{
    //=================================================
    // join_cols
    Eigen::MatrixXd X = S.transpose()*A.transpose();
    Eigen::MatrixXd N = Ns.transpose();
    Eigen::MatrixXd Y(X.rows(),X.cols()+N.cols());
    Y.leftCols(X.cols()) = X;
    Y.rightCols(N.cols()) = N;
    //=================================================

    //std::cout << "A:" << std::endl << A << std::endl;
    //std::cout << "S:" << std::endl << S << std::endl;
    //std::cout << "Ns:" << std::endl << Ns << std::endl;
    //std::cout << "X:" << std::endl << X << std::endl;
    //std::cout << "N:" << std::endl << N << std::endl;
    //std::cout << "Y:" << std::endl << Y << std::endl;

    //=================================================
    // qr decomposition
    Eigen::HouseholderQR<Eigen::MatrixXd> qr;
    qr.compute(Y);

    Eigen::MatrixXd R = qr.matrixQR();
    Eigen::MatrixXd Q = qr.householderQ();
    //=================================================

    //std::cout << "R:" << std::endl << R << std::endl;
    //std::cout << "Q:" << std::endl << Q << std::endl;

    //std::cout << "Q*R:" << std::endl << Q*R << std::endl;

    //Eigen::LLT

    //Eigen::MatrixXd D = Y, Q, R;
    //qr_econ(Q, R, D);
    return R.transpose();
}

inline Eigen::MatrixXd qrFactor_A(const Eigen::MatrixXd& A1,
                                  const Eigen::MatrixXd& S1,
                                  const Eigen::MatrixXd& Ns1)
{
    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    arma::Mat<double> A = EA(A1);
    arma::Mat<double> S = EA(S1);
    arma::Mat<double> Ns = EA(Ns1);
    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    arma::Mat<double> D = join_cols(S.t()*A.t(),Ns.t()), Q, R;
    qr_econ(Q, R, D);
    arma::Mat<double> RT = R.t();
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    return AE(RT);
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
}

inline Eigen::MatrixXd ComputeKalmanGain_A(const Eigen::MatrixXd& Sy1,
                                           const Eigen::MatrixXd& Pxy1)
{
    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    arma::Mat<double> Sy = EA(Sy1);
    arma::Mat<double> Pxy = EA(Pxy1);
    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    arma::Mat<double> K1 = arma::solve(arma::trimatl(Sy), trans(Pxy), arma::solve_opts::fast);
    arma::Mat<double> K  = arma::trans(arma::solve(arma::trimatu(trans(Sy)), K1, arma::solve_opts::fast));

    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    return AE(K);
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
}

inline arma::mat svdPSD(const arma::mat &A) {
    arma::mat U,V;
    arma::vec s;
    svd(U,s,V,A);
    return V*sqrt(diagmat(s));
}

inline Eigen::MatrixXd cholPSD_A(const Eigen::MatrixXd& A1)
{
    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    arma::Mat<double> A = EA(A1);
    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    arma::Mat<double> ret;
    if (!chol(ret, A)) {
        return AE(svdPSD(A));
    }
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    return AE(ret.t());
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
}

inline double ComputeAngleDifference(double a1, double a2) {
    return std::arg(std::complex<double>(cos(a1 - a2), sin(a1 - a2)));
}

}
