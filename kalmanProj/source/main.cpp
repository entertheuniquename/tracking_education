#include <iostream>

#include "kf_armadillo.h"
#include "kf_eigen3.h"

#include<QApplication>
#include<QChartView>
#include<QLineSeries>
#include<QScatterSeries>

#include "data_creator_eigen3.h"
#include "printer_eigen3.h"

#include<unsupported/Eigen/MatrixFunctions>
#include<Eigen/Eigenvalues>

using namespace QtCharts;

int multix(const Eigen::MatrixXd a,
           const Eigen::MatrixXd b,
                 Eigen::MatrixXd& result)
{
    if(a.cols()==b.cols() && a.rows()==b.rows())
    {
        result.resize(a.rows(),a.cols());
        for(int k1=0;k1<a.rows();k1++)
            for(int k2=0;k2<a.cols();k2++)
                result(k1,k2) = a(k1,k2)*b(k1,k2);
        return 0;
    }
    else
        return 1;
};


Eigen::MatrixXd estimator_step(const Eigen::MatrixXd& measurements,
                               const Eigen::MatrixXd& P0,
                               const Eigen::MatrixXd& A,
                               const Eigen::MatrixXd& H,
                               const Eigen::MatrixXd& B,
                               const Eigen::MatrixXd& u,
                               const Eigen::MatrixXd& Q,
                               const Eigen::MatrixXd& G,
                               const Eigen::MatrixXd& R)
{
    KFE* kfe = new KFE(measurements.col(1),P0,A,Q,G,H,R);
    Eigen::MatrixXd estimations(measurements.rows(),measurements.cols());
    estimations.setZero();
    for(int i=0;i<measurements.cols();i++)
    {
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> pred = kfe->predict(A,H,u,B);
        Eigen::MatrixXd zi(3,1);
        zi << measurements(0,i), measurements(2,i), measurements(4,i);
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> corr = kfe->correct(H,zi);
        estimations.col(i) = corr.first;
    }
    return estimations;
};

Eigen::MatrixXd estimator_errors(const Eigen::MatrixXd& measurements,
                                 const Eigen::MatrixXd& estimations)
{
    Eigen::MatrixXd errors(estimations.rows(),estimations.cols());
    errors.setZero();
    multix(estimations-measurements,estimations-measurements,errors);
    return errors;
};

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    // == make input data ==
    double meas_var = 4.;
    double max_speed = 4.;
    double process_var = 1.;
    double dt = 0.2;
    Eigen::MatrixXd A(6,6);
    A << 1., dt, 0., 0., 0., 0.,
          0., 1., 0., 0., 0., 0.,
          0., 0., 1., dt, 0., 0.,
          0., 0., 0., 1., 0., 0.,
          0., 0., 0., 0., 1., dt,
          0., 0., 0., 0., 0., 1.;
    Eigen::MatrixXd H(3,6);
    H << 1., 0., 0., 0., 0., 0.,
          0., 0., 1., 0., 0., 0.,
          0., 0., 0., 0., 1., 0.;
    Eigen::MatrixXd R(3,3);
    R << meas_var,      0.,       0.,
                0.,meas_var,       0.,
                0.,      0., meas_var;
    Eigen::MatrixXd Q(3,3);
    Q << process_var,         0.,          0.,
                   0.,process_var,          0.,
                   0.,         0., process_var;
    Eigen::MatrixXd G(6,3);
    G << dt*dt/2.,       0.,       0.,
                dt,       0.,       0.,
                0., dt*dt/2.,       0.,
                0.,       dt,       0.,
                0.,       0., dt*dt/2.,
                0.,       0.,       dt;
    Eigen::MatrixXd x0(1,6);
    x0 << 10., 2., 0., 0., 0., 0.;
    Eigen::MatrixXd P0(6,6);
    P0 << meas_var,                          0.,       0.,                         0.,       0.,                         0.,
                0., (max_speed/3)*(max_speed/3),       0.,                         0.,       0.,                         0.,
                0.,                          0., meas_var,                         0.,       0.,                         0.,
                0.,                          0.,       0.,(max_speed/3)*(max_speed/3),       0.,                         0.,
                0.,                          0.,       0.,                         0., meas_var,                         0.,
                0.,                          0.,       0.,                         0.,       0.,(max_speed/3)*(max_speed/3);
    Eigen::MatrixXd B(6,6);
    B.setZero();
    Eigen::MatrixXd u(6,1);
    u.setZero();

    // == statistic ==
    int iterations_statistic = 2000;

    Eigen::MatrixXd var_err;

    for(int i=0;i<iterations_statistic;i++)
    {
        // == make measurements ==
        std::pair<Eigen::MatrixXd,Eigen::MatrixXd> out = make_data(x0,A,G*Q*G.transpose(),100,-1.,1.);
        if(i==0)
        {
            var_err.resize(out.second.rows(),out.second.cols());
            var_err.setZero();
        }
        // == estimation ==
        Eigen::MatrixXd est = estimator_step(out.second,P0,A,H,B,u,Q,G,R);
        Eigen::MatrixXd err = estimator_errors(out.second,est);
        if(i==0)
            print_chart(out.first,out.second,est,err);
        var_err+=err;
    }

    var_err/=iterations_statistic;
    print_chart_statistic(var_err);

    return a.exec();
}
