#include "../source/tracker_prototype.h"
#include "../source/gnn_prototype.h"
#include "../source/kf.h"
#include "../source/models.h"
#include <gtest/gtest.h>

TEST (Tracker_prototype,tracker_base_test) {
    //----------------------------------------------------------------------
    ASSERT_TRUE(true);
    //----------------------------------------------------------------------
    struct stateModel
    {
        Eigen::MatrixXd operator()(const Eigen::MatrixXd& x,double T)
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
        Eigen::MatrixXd operator()(const Eigen::MatrixXd& x)
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
        Eigen::MatrixXd operator()(double T)
        {
            Eigen::MatrixXd G(4,2);
            G <<   T*T/2.,       0.,
                       T ,       0.,
                       0.,   T*T/2.,
                       0.,       T ;
            return G;
        }
    };
    double dt = 0.2;
    //----------------------------------------------------------------------
    Tracker::Tracker_prototype<Eigen::MatrixXd,
                               Tracker::EstimatorInitializator4<Eigen::MatrixXd,
                                                                Estimator::KF<Eigen::MatrixXd,
                                                                              stateModel,
                                                                              measureModel,
                                                                              noiseTransitionModel>,
                                                                Tracker::Measurement2<Eigen::MatrixXd>>,
                               Estimator::KF<Eigen::MatrixXd,
                                             stateModel,
                                             measureModel,
                                             noiseTransitionModel>,
                               Tracker::Track<Eigen::MatrixXd,
                                              Tracker::Measurement2<Eigen::MatrixXd>,
                               Tracker::EstimatorInitializator4<Eigen::MatrixXd,
                                                                Estimator::KF<Eigen::MatrixXd,
                                                                              stateModel,
                                                                              measureModel,
                                                                              noiseTransitionModel>,
                                                                              Tracker::Measurement2<Eigen::MatrixXd>>>,
                               Tracker::Measurement2<Eigen::MatrixXd>> tracker;

    Tracker::Measurement2<Eigen::MatrixXd> initMeasurement1{0.,100000.,100000.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> initMeasurement2{0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> initMeasurement3{0.,100000.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> initMeasurement4{0.,0.,100000.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};

    std::vector<Tracker::Measurement2<Eigen::MatrixXd>> measurements;
    measurements.push_back(initMeasurement1);
    measurements.push_back(initMeasurement2);
    measurements.push_back(initMeasurement3);
    measurements.push_back(initMeasurement4);

    tracker.step(measurements);
    measurements.clear();

    Tracker::Measurement2<Eigen::MatrixXd> step1Measurement1{0.,99990.,99990.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> step1Measurement2{0.,10.,10.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> step1Measurement3{0.,99990.,10.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> step1Measurement4{0.,10.,99990.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};

    measurements.push_back(step1Measurement1);
    measurements.push_back(step1Measurement2);
    measurements.push_back(step1Measurement3);
    measurements.push_back(step1Measurement4);

    tracker.step(measurements);
    measurements.clear();

    Tracker::Measurement2<Eigen::MatrixXd> step2Measurement1{0.,99980.,99980.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> step2Measurement2{0.,20.,20.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> step2Measurement3{0.,99980.,20.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> step2Measurement4{0.,20.,99980.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};

    measurements.push_back(step2Measurement1);
    measurements.push_back(step2Measurement2);
    measurements.push_back(step2Measurement3);
    measurements.push_back(step2Measurement4);

    tracker.step(measurements);
    measurements.clear();

    //----------------------------------------------------------------------
    //ASSERT_TRUE(controlState1.isApprox(tracks.at(0).getState(),0.00001));
    //ASSERT_TRUE(controlCovariance1.isApprox(tracks.at(0).getCovariance(),0.00001));
    //ASSERT_TRUE(controlState2.isApprox(tracks.at(1).getState(),0.00001));
    //ASSERT_TRUE(controlCovariance2.isApprox(tracks.at(1).getCovariance(),0.00001));
    //ASSERT_TRUE(controlState3.isApprox(tracks.at(2).getState(),0.00001));
    //ASSERT_TRUE(controlCovariance3.isApprox(tracks.at(2).getCovariance(),0.00001));
}
