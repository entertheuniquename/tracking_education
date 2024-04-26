#include "../source/ekf_eigen3.h"
#include "../source/track.h"
#include <iostream>

int main(int argc, char *argv[])
{
    /////////////////////////////////////////////////////////
    /// MEASUREMENT
    Measurement<Eigen::MatrixXd> measurement;
    measurement.timepoint = 1111.1;
    measurement.point.resize(3,1);
    measurement.point << 1., 2., 3.;
    //measurement.measurement_noise.resize(3,3);
    //measurement.measurement_noise << 1., 2., 3.,
    //                                 4., 5., 6.,
    //                                 7., 8., 9.;
    /////////////////////////////////////////////////////////
    /// ESTIMATOR_INIT
    EstimatorInitKFE<Eigen::MatrixXd,Models::StateModelA<Eigen::MatrixXd>,Models::MeasureModelA<Eigen::MatrixXd>> eikfe(measurement);
    std::cout << "eikfe.SM:" << std::endl << eikfe.SM << std::endl;
    std::cout << "eikfe.MM:" << std::endl << eikfe.MM << std::endl;
    std::cout << "eikfe.GM:" << std::endl << eikfe.GM << std::endl;
    std::cout << "eikfe.R:" << std::endl << eikfe.R << std::endl;
    std::cout << "eikfe.Q:" << std::endl << eikfe.Q << std::endl;
    std::cout << "eikfe.P0:" << std::endl << eikfe.P0 << std::endl;
    std::cout << "eikfe.x0:" << std::endl << eikfe.x0 << std::endl;

    EstimatorInitEKFE<Eigen::MatrixXd,Models::StateModelZ<Eigen::MatrixXd>,Models::MeasureModelZ<Eigen::MatrixXd>> eiekfe(measurement);
    std::cout << "eiekfe.R:" << std::endl << eiekfe.R << std::endl;
    std::cout << "eiekfe.Q:" << std::endl << eiekfe.Q << std::endl;
    std::cout << "eiekfe.P0:" << std::endl << eiekfe.P0 << std::endl;
    std::cout << "eiekfe.x0:" << std::endl << eiekfe.x0 << std::endl;
    /////////////////////////////////////////////////////////
    /// TRACK
    Track<Eigen::MatrixXd,
          Estimator::KFE<Eigen::MatrixXd,Models::StateModelA<Eigen::MatrixXd>,Models::MeasureModelA<Eigen::MatrixXd>>,
          EstimatorInitKFE<Eigen::MatrixXd,Models::StateModelA<Eigen::MatrixXd>,Models::MeasureModelA<Eigen::MatrixXd>>,
          Models::StateModelA<Eigen::MatrixXd>,Models::MeasureModelA<Eigen::MatrixXd>>
            tkfe(measurement);
    tkfe.step(measurement);
    tkfe.step(6.);

    Track<Eigen::MatrixXd,
          Estimator::EKFE<Eigen::MatrixXd,Models::StateModelZ<Eigen::MatrixXd>,Models::MeasureModelZ<Eigen::MatrixXd>>,
          EstimatorInitEKFE<Eigen::MatrixXd,Models::StateModelZ<Eigen::MatrixXd>,Models::MeasureModelZ<Eigen::MatrixXd>>,
          Models::StateModelZ<Eigen::MatrixXd>,Models::MeasureModelZ<Eigen::MatrixXd>>
            tekfe(measurement);
    tekfe.step(measurement);
    tekfe.step(6.);
    /////////////////////////////////////////////////////////
}
