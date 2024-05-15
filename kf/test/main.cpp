#include "../source/ekf_eigen3.h"
#include "../source/track.h"
#include <iostream>

int main(int argc, char *argv[])
{
    /////////////////////////////////////////////////////////
    /// MEASUREMENT
    Measurement<Eigen::MatrixXd> measurement;
//    measurement.timepoint = 1111.1;
//    measurement.point.resize(3,1);
//    measurement.point << 1., 2., 3.;
    //measurement.measurement_noise.resize(3,3);
    //measurement.measurement_noise << 1., 2., 3.,
    //                                 4., 5., 6.,
    //                                 7., 8., 9.;
    /////////////////////////////////////////////////////////
    /// ESTIMATOR_INIT
//    EstimatorInitKFE<Eigen::MatrixXd,Models::StateModel_CV<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZ_XYZ<Eigen::MatrixXd>> eikfe(measurement);
//    std::cout << "eikfe.SM:" << std::endl << eikfe.SM << std::endl;
//    std::cout << "eikfe.MM:" << std::endl << eikfe.MM << std::endl;
//    std::cout << "eikfe.GM:" << std::endl << eikfe.GM << std::endl;
//    std::cout << "eikfe.R:" << std::endl << eikfe.R << std::endl;
//    std::cout << "eikfe.Q:" << std::endl << eikfe.Q << std::endl;
//    std::cout << "eikfe.P0:" << std::endl << eikfe.P0 << std::endl;
//    std::cout << "eikfe.x0:" << std::endl << eikfe.x0 << std::endl;

//    EstimatorInitEKFE<Eigen::MatrixXd,Models::StateModel_CV<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZ_EAR<Eigen::MatrixXd>> eiekfe(measurement);
//    std::cout << "eiekfe.R:" << std::endl << eiekfe.R << std::endl;
//    std::cout << "eiekfe.Q:" << std::endl << eiekfe.Q << std::endl;
//    std::cout << "eiekfe.P0:" << std::endl << eiekfe.P0 << std::endl;
//    std::cout << "eiekfe.x0:" << std::endl << eiekfe.x0 << std::endl;
    /////////////////////////////////////////////////////////
    /// TRACK
//    Track<Eigen::MatrixXd,
//          Estimator::KFE<Eigen::MatrixXd,Models::StateModel_CV<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZ_XYZ<Eigen::MatrixXd>>,
//          EstimatorInitKFE<Eigen::MatrixXd,Models::StateModel_CV<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZ_XYZ<Eigen::MatrixXd>>,
//          Models::StateModel_CV<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZ_XYZ<Eigen::MatrixXd>>
//            tkfe(measurement);
//    tkfe.step(measurement);
//    tkfe.step(6.);

//    Track<Eigen::MatrixXd,
//          Estimator::EKFE<Eigen::MatrixXd,Models::StateModel_CV<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZ_EAR<Eigen::MatrixXd>>,
//          EstimatorInitEKFE<Eigen::MatrixXd,Models::StateModel_CV<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZ_EAR<Eigen::MatrixXd>>,
//          Models::StateModel_CV<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZ_EAR<Eigen::MatrixXd>>
//            tekfe(measurement);
//    tekfe.step(measurement);
//    tekfe.step(6.);
    /////////////////////////////////////////////////////////
//    Eigen::MatrixXd a0(7,1);
//    a0 << 6,5,4,3,2,1,0;
//    std::cout << "a0: " << a0 << std::endl;
//    Eigen::MatrixXd a1;
//    Models::StateModel_CT<Eigen::MatrixXd> m1;
//    a1 = m1(a0,6.);
//    std::cout << "a1: " << a1 << std::endl;
    /////////////////////////////////////////////////////////
//    double a60 = 60.;
//    double a45 = 45.;
//    double a30 = 30.;
//    double a0 = 0.;

//    std::cout << "std::sin(45(d)) = " << std::sin(a45) << std::endl;
//    std::cout << "std::sin(45(r)) = " << std::sin(Utils::deg2rad(a45)) << std::endl;
    /////////////////////////////////////////////////////////
//    Eigen::MatrixXd A(6,6);
//    A << 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
//         2.1, 2.2, 2.3, 2.4, 2.5, 2.6,
//         3.1, 3.2, 3.3, 3.4, 3.5, 3.6,
//         4.1, 4.2, 4.3, 4.4, 4.5, 4.6,
//         5.1, 5.2, 5.3, 5.4, 5.5, 5.6,
//         6.1, 6.2, 6.3, 6.4, 6.5, 6.6;
//    std::cout << "A:" << std::endl << A << std::endl;

//    Eigen::MatrixXd P(6,6);
//    P <<  1.,  2.,  3.,  4.,  5.,  6.,
//          7.,  8.,  9., 10., 11., 12.,
//         13., 14., 15., 16., 17., 18.,
//         19., 20., 21., 22., 23., 24.,
//         25., 26., 27., 28., 29., 30.,
//         31., 32., 33., 34., 35., 36.;
//    std::cout << "P:" << std::endl << P << std::endl;

//    Eigen::MatrixXd AP = A*P;
//    std::cout << "AP:" << std::endl << AP << std::endl;

//    Eigen::MatrixXd APAt = A*P*A.transpose();
//    std::cout << "APAt:" << std::endl << APAt << std::endl;

//    Eigen::MatrixXd APAt_ = A*(A*P).transpose();
//    std::cout << "APAt_:" << std::endl << APAt_ << std::endl;

//    Eigen::MatrixXd APAt__ = (A*(A*P).transpose()).transpose();
//    std::cout << "APAt__:" << std::endl << APAt__ << std::endl;

//    if(APAt==APAt__)
//        std::cout << "good!" << std::endl;
//    else
//        std::cout << "bad!" << std::endl;
    /////////////////////////////////////////////////////////
}
