#include "../source/tracker.h"
#include "../source/gnn.h"
#include "../source/kf.h"
#include "../source/models.h"
#include <gtest/gtest.h>

TEST (TrackerGNN,tracker_base_test) {
    //----------------------------------------------------------------------
    ASSERT_TRUE(true);
    //----------------------------------------------------------------------
    struct stateModel
    {
        Eigen::MatrixXd operator()(const Eigen::MatrixXd& x,double T, MatrixXd state=MatrixXd{})
        {
            Eigen::MatrixXd F(4,4);
            F << 1., T , 0., 0.,
                 0., 1., 0., 0.,
                 0., 0., 1., T ,
                 0., 0., 0., 1.;
            return F*x;
        }
    };
    struct measureModel
    {
        Eigen::MatrixXd operator()(const Eigen::MatrixXd& x, Eigen::MatrixXd z = Eigen::MatrixXd{}, Eigen::MatrixXd state = Eigen::MatrixXd{})
        {
            Eigen::MatrixXd H(2,4);
            H << 1., 0., 0., 0.,
                 0., 0., 1., 0.;
            return H*x;
        }
        Eigen::MatrixXd operator()()
        {
            Eigen::MatrixXd H(2,4);
            H << 1., 0., 0., 0.,
                 0., 0., 1., 0.;
            return H;
        }
    };
    struct noiseTransitionModel
    {
        Eigen::MatrixXd matrix(double T)
        {
            Eigen::MatrixXd G(4,2);
            G <<   T*T/2.,       0.,
                       T ,       0.,
                       0.,   T*T/2.,
                       0.,       T ;
            return G;
        }
        Eigen::MatrixXd operator()(const Eigen::MatrixXd& x, double T)
        {
            Eigen::MatrixXd G(4,2);
            G <<   T*T/2.,       0.,
                       T ,       0.,
                       0.,   T*T/2.,
                       0.,       T ;
            return G*x;
        }
    };
    double dt = 0.2;
    //----------------------------------------------------------------------
    using MatrixType = Eigen::MatrixXd;
    using EstimatorType = Estimator::KF<MatrixType,stateModel,measureModel,noiseTransitionModel>;
    using MeasurementType = Tracker::Measurement2<MatrixType>;
    using EstimatorInitializatorType = Tracker::EstimatorInitializator4<MatrixType,EstimatorType,MeasurementType>;
    using TrackType = Tracker::Track<MatrixType,MeasurementType,EstimatorInitializatorType>;
    //----------------------------------------------------------------------
    Tracker::TrackerGNN<MatrixType,
                               EstimatorInitializatorType,
                               EstimatorType,
                               TrackType,
                               MeasurementType> tracker;

    MeasurementType initMeasurement1{0,0.,100000.,100000.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    MeasurementType initMeasurement2{0,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    MeasurementType initMeasurement3{0,0.,100000.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    MeasurementType initMeasurement4{0,0.,0.,100000.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};

    std::vector<MeasurementType> measurements;
    measurements.push_back(initMeasurement1);
    measurements.push_back(initMeasurement2);
    measurements.push_back(initMeasurement3);
    measurements.push_back(initMeasurement4);

    tracker.step(measurements,dt);
    measurements.clear();

    MeasurementType step1Measurement1{0,initMeasurement1.timepoint()+dt,99990.,99990.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    MeasurementType step1Measurement2{0,initMeasurement2.timepoint()+dt,10.,10.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    MeasurementType step1Measurement3{0,initMeasurement3.timepoint()+dt,99990.,10.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    MeasurementType step1Measurement4{0,initMeasurement4.timepoint()+dt,10.,99990.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};

    measurements.push_back(step1Measurement1);
    measurements.push_back(step1Measurement2);
    measurements.push_back(step1Measurement3);
    measurements.push_back(step1Measurement4);

    tracker.step(measurements,dt);
    measurements.clear();

    MeasurementType step2Measurement1{0,step1Measurement1.timepoint()+dt,99980.,99980.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    MeasurementType step2Measurement2{0,step1Measurement2.timepoint()+dt,20.,20.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    MeasurementType step2Measurement3{0,step1Measurement3.timepoint()+dt,99980.,20.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    MeasurementType step2Measurement4{0,step1Measurement4.timepoint()+dt,20.,99980.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};

    measurements.push_back(step2Measurement1);
    measurements.push_back(step2Measurement2);
    measurements.push_back(step2Measurement3);
    measurements.push_back(step2Measurement4);

    tracker.step(measurements,dt);
    measurements.clear();
    //----------------------------------------------------------------------
    TrackType track1;
    track1.initialization(initMeasurement1);
    track1.step(step1Measurement1);
    auto a1 = track1.step(step2Measurement1);

    TrackType track2;
    track2.initialization(initMeasurement2);
    track2.step(step1Measurement2);
    auto a2 = track2.step(step2Measurement2);

    TrackType track3;
    track3.initialization(initMeasurement3);
    track3.step(step1Measurement3);
    auto a3 = track3.step(step2Measurement3);
    //----------------------------------------------------------------------
    ASSERT_TRUE(a1.first.isApprox(tracker.tracks.at(0)->getState(),0.00001));
    ASSERT_TRUE(a1.second.isApprox(tracker.tracks.at(0)->getCovariance(),0.00001));
    ASSERT_TRUE(a2.first.isApprox(tracker.tracks.at(1)->getState(),0.00001));
    ASSERT_TRUE(a2.second.isApprox(tracker.tracks.at(1)->getCovariance(),0.00001));
    ASSERT_TRUE(a3.first.isApprox(tracker.tracks.at(2)->getState(),0.00001));
    ASSERT_TRUE(a3.second.isApprox(tracker.tracks.at(2)->getCovariance(),0.00001));
}
