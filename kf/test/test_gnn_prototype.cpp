#include "../source/gnn_prototype.h".h"
#include "../source/kf.h"
#include <gtest/gtest.h>

TEST (GNN_prototype,gnn_base3x3_test) {
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
    Tracker::Track<Eigen::MatrixXd,Tracker::Measurement2<Eigen::MatrixXd>,Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>> track1;
    Tracker::Track<Eigen::MatrixXd,Tracker::Measurement2<Eigen::MatrixXd>,Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>> controlTrack1;

    Tracker::Track<Eigen::MatrixXd,Tracker::Measurement2<Eigen::MatrixXd>,Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>> track2;
    Tracker::Track<Eigen::MatrixXd,Tracker::Measurement2<Eigen::MatrixXd>,Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>> controlTrack2;

    Tracker::Track<Eigen::MatrixXd,Tracker::Measurement2<Eigen::MatrixXd>,Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>> track3;
    Tracker::Track<Eigen::MatrixXd,Tracker::Measurement2<Eigen::MatrixXd>,Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>> controlTrack3;

    ASSERT_TRUE(track1.isInit()==false);
    ASSERT_TRUE(track2.isInit()==false);
    ASSERT_TRUE(track3.isInit()==false);
    ASSERT_TRUE(controlTrack1.isInit()==false);
    ASSERT_TRUE(controlTrack2.isInit()==false);
    ASSERT_TRUE(controlTrack3.isInit()==false);

    Tracker::Measurement2<Eigen::MatrixXd> initMeasurement1{0.,100.,100.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> initMeasurement2{0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> initMeasurement3{0.,-100.,-100.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};

    Tracker::Measurement2<Eigen::MatrixXd> stepMeasurement1{initMeasurement1.timepoint()+dt,110.,110.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> stepMeasurement2{initMeasurement2.timepoint()+dt,20.,20.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> stepMeasurement3{initMeasurement3.timepoint()+dt,-70.,-70.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,};

    track1.initialization<Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>>
            (initMeasurement1);
    controlTrack1.initialization<Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>>
            (initMeasurement1);

    track2.initialization<Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>>
            (initMeasurement2);
    controlTrack2.initialization<Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>>
            (initMeasurement2);

    track3.initialization<Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>>
            (initMeasurement3);
    controlTrack3.initialization<Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>>
            (initMeasurement3);

    ASSERT_TRUE(track1.isInit()==true);
    ASSERT_TRUE(track2.isInit()==true);
    ASSERT_TRUE(track3.isInit()==true);
    ASSERT_TRUE(controlTrack1.isInit()==true);
    ASSERT_TRUE(controlTrack2.isInit()==true);
    ASSERT_TRUE(controlTrack3.isInit()==true);

    //----------------------------------------------------------------------
    auto controlTrack1_res = controlTrack1.step(stepMeasurement1);
    Eigen::MatrixXd controlState1 = controlTrack1_res.first;
    Eigen::MatrixXd controlCovariance1 = controlTrack1_res.second;

    auto controlTrack2_res = controlTrack2.step(stepMeasurement2);
    Eigen::MatrixXd controlState2 = controlTrack2_res.first;
    Eigen::MatrixXd controlCovariance2 = controlTrack2_res.second;

    auto controlTrack3_res = controlTrack3.step(stepMeasurement3);
    Eigen::MatrixXd controlState3 = controlTrack3_res.first;
    Eigen::MatrixXd controlCovariance3 = controlTrack3_res.second;
    //----------------------------------------------------------------------
    std::vector<Tracker::Track<Eigen::MatrixXd,Tracker::Measurement2<Eigen::MatrixXd>,Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>>> tracks;
    tracks.push_back(track1);
    tracks.push_back(track2);
    tracks.push_back(track3);

    std::vector<Tracker::Measurement2<Eigen::MatrixXd>> measurements;
    measurements.push_back(stepMeasurement1);
    measurements.push_back(stepMeasurement2);
    measurements.push_back(stepMeasurement3);
    //----------------------------------------------------------------------
    std::vector<Eigen::MatrixXd> vec_zp;
    vec_zp.resize(tracks.size());
    std::vector<Eigen::MatrixXd> vec_S;
    vec_S.resize(tracks.size());
    std::vector<Eigen::MatrixXd> vec_z;
    vec_z.resize(measurements.size());

    for(int i=0;i<tracks.size();i++)
    {
        auto t0 = tracks.at(i).getMeasurementPredictData(dt);
        vec_zp[i] = t0.first;
        vec_S[i] = t0.second;
    }
    for(int i=0;i<measurements.size();i++)
        vec_z[i] = measurements.at(i).get();

    //----------------------------------------------------------------------
    Eigen::MatrixXd association_matrix = Association::GNN_prototype<Eigen::MatrixXd>()(vec_z,vec_zp,vec_S);
    std::vector<std::pair<int,int>> result_pairs = Association::Auction_prototype<Eigen::MatrixXd>()(association_matrix/*,vec_prices*/);
    //----------------------------------------------------------------------

    for(int i=0;i<result_pairs.size();i++)
        tracks.at(result_pairs.at(i).first).step(measurements.at(result_pairs.at(i).second));

    //----------------------------------------------------------------------
    ASSERT_TRUE(controlState1.isApprox(tracks.at(0).getState(),0.00001));
    ASSERT_TRUE(controlCovariance1.isApprox(tracks.at(0).getCovariance(),0.00001));
    ASSERT_TRUE(controlState2.isApprox(tracks.at(1).getState(),0.00001));
    ASSERT_TRUE(controlCovariance2.isApprox(tracks.at(1).getCovariance(),0.00001));
    ASSERT_TRUE(controlState3.isApprox(tracks.at(2).getState(),0.00001));
    ASSERT_TRUE(controlCovariance3.isApprox(tracks.at(2).getCovariance(),0.00001));
}

TEST (GNN_prototype,gnn_base3x5_test) {
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
    Tracker::Track<Eigen::MatrixXd,Tracker::Measurement2<Eigen::MatrixXd>,Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>> track1;
    Tracker::Track<Eigen::MatrixXd,Tracker::Measurement2<Eigen::MatrixXd>,Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>> controlTrack1;

    Tracker::Track<Eigen::MatrixXd,Tracker::Measurement2<Eigen::MatrixXd>,Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>> track2;
    Tracker::Track<Eigen::MatrixXd,Tracker::Measurement2<Eigen::MatrixXd>,Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>> controlTrack2;

    Tracker::Track<Eigen::MatrixXd,Tracker::Measurement2<Eigen::MatrixXd>,Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>> track3;
    Tracker::Track<Eigen::MatrixXd,Tracker::Measurement2<Eigen::MatrixXd>,Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>> controlTrack3;

    ASSERT_TRUE(track1.isInit()==false);
    ASSERT_TRUE(track2.isInit()==false);
    ASSERT_TRUE(track3.isInit()==false);
    ASSERT_TRUE(controlTrack1.isInit()==false);
    ASSERT_TRUE(controlTrack2.isInit()==false);
    ASSERT_TRUE(controlTrack3.isInit()==false);

    Tracker::Measurement2<Eigen::MatrixXd> initMeasurement1{0.,100.,100.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> initMeasurement2{0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> initMeasurement3{0.,-100.,-100.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};

    Tracker::Measurement2<Eigen::MatrixXd> stepMeasurement1{initMeasurement1.timepoint()+dt,110.,110.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> stepMeasurement2{initMeasurement2.timepoint()+dt,20.,20.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> stepMeasurement3{initMeasurement3.timepoint()+dt,-70.,-70.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> stepMeasurement4{initMeasurement1.timepoint()+dt,220.,220.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> stepMeasurement5{initMeasurement2.timepoint()+dt,-220.,-220.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,};


    track1.initialization<Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>>
            (initMeasurement1);
    controlTrack1.initialization<Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>>
            (initMeasurement1);

    track2.initialization<Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>>
            (initMeasurement2);
    controlTrack2.initialization<Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>>
            (initMeasurement2);

    track3.initialization<Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>>
            (initMeasurement3);
    controlTrack3.initialization<Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>>
            (initMeasurement3);

    ASSERT_TRUE(track1.isInit()==true);
    ASSERT_TRUE(track2.isInit()==true);
    ASSERT_TRUE(track3.isInit()==true);
    ASSERT_TRUE(controlTrack1.isInit()==true);
    ASSERT_TRUE(controlTrack2.isInit()==true);
    ASSERT_TRUE(controlTrack3.isInit()==true);

    //----------------------------------------------------------------------
    auto controlTrack1_res = controlTrack1.step(stepMeasurement1);
    Eigen::MatrixXd controlState1 = controlTrack1_res.first;
    Eigen::MatrixXd controlCovariance1 = controlTrack1_res.second;

    auto controlTrack2_res = controlTrack2.step(stepMeasurement2);
    Eigen::MatrixXd controlState2 = controlTrack2_res.first;
    Eigen::MatrixXd controlCovariance2 = controlTrack2_res.second;

    auto controlTrack3_res = controlTrack3.step(stepMeasurement3);
    Eigen::MatrixXd controlState3 = controlTrack3_res.first;
    Eigen::MatrixXd controlCovariance3 = controlTrack3_res.second;
    //----------------------------------------------------------------------
    std::vector<Tracker::Track<Eigen::MatrixXd,Tracker::Measurement2<Eigen::MatrixXd>,Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>>> tracks;
    tracks.push_back(track1);
    tracks.push_back(track2);
    tracks.push_back(track3);

    std::vector<Tracker::Measurement2<Eigen::MatrixXd>> measurements;
    measurements.push_back(stepMeasurement1);
    measurements.push_back(stepMeasurement2);
    measurements.push_back(stepMeasurement3);
    measurements.push_back(stepMeasurement4);
    measurements.push_back(stepMeasurement5);
    //----------------------------------------------------------------------
    std::vector<Eigen::MatrixXd> vec_zp;
    vec_zp.resize(tracks.size());
    std::vector<Eigen::MatrixXd> vec_S;
    vec_S.resize(tracks.size());
    std::vector<Eigen::MatrixXd> vec_z;
    vec_z.resize(measurements.size());

    for(int i=0;i<tracks.size();i++)
    {
        auto t0 = tracks.at(i).getMeasurementPredictData(dt);
        vec_zp[i] = t0.first;
        vec_S[i] = t0.second;
    }
    for(int i=0;i<measurements.size();i++)
        vec_z[i] = measurements.at(i).get();

    //----------------------------------------------------------------------
    Eigen::MatrixXd association_matrix = Association::GNN_prototype<Eigen::MatrixXd>()(vec_z,vec_zp,vec_S);
    std::vector<std::pair<int,int>> result_pairs = Association::Auction_prototype<Eigen::MatrixXd>()(association_matrix/*,vec_prices*/);
    //----------------------------------------------------------------------

    for(int i=0;i<result_pairs.size();i++)
        tracks.at(result_pairs.at(i).first).step(measurements.at(result_pairs.at(i).second));

    //----------------------------------------------------------------------
    ASSERT_TRUE(controlState1.isApprox(tracks.at(0).getState(),0.00001));
    ASSERT_TRUE(controlCovariance1.isApprox(tracks.at(0).getCovariance(),0.00001));
    ASSERT_TRUE(controlState2.isApprox(tracks.at(1).getState(),0.00001));
    ASSERT_TRUE(controlCovariance2.isApprox(tracks.at(1).getCovariance(),0.00001));
    ASSERT_TRUE(controlState3.isApprox(tracks.at(2).getState(),0.00001));
    ASSERT_TRUE(controlCovariance3.isApprox(tracks.at(2).getCovariance(),0.00001));
}

TEST (GNN_prototype_2,gnn_base3x5_test) {
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
    Tracker::Track<Eigen::MatrixXd,Tracker::Measurement2<Eigen::MatrixXd>,Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>> track1;
    Tracker::Track<Eigen::MatrixXd,Tracker::Measurement2<Eigen::MatrixXd>,Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>> controlTrack1;

    Tracker::Track<Eigen::MatrixXd,Tracker::Measurement2<Eigen::MatrixXd>,Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>> track2;
    Tracker::Track<Eigen::MatrixXd,Tracker::Measurement2<Eigen::MatrixXd>,Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>> controlTrack2;

    Tracker::Track<Eigen::MatrixXd,Tracker::Measurement2<Eigen::MatrixXd>,Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>> track3;
    Tracker::Track<Eigen::MatrixXd,Tracker::Measurement2<Eigen::MatrixXd>,Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>> controlTrack3;

    ASSERT_TRUE(track1.isInit()==false);
    ASSERT_TRUE(track2.isInit()==false);
    ASSERT_TRUE(track3.isInit()==false);
    ASSERT_TRUE(controlTrack1.isInit()==false);
    ASSERT_TRUE(controlTrack2.isInit()==false);
    ASSERT_TRUE(controlTrack3.isInit()==false);

    Tracker::Measurement2<Eigen::MatrixXd> initMeasurement1{0.,100.,100.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> initMeasurement2{0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> initMeasurement3{0.,-100.,-100.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};

    Tracker::Measurement2<Eigen::MatrixXd> stepMeasurement1{initMeasurement1.timepoint()+dt,110.,110.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> stepMeasurement2{initMeasurement2.timepoint()+dt,20.,20.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> stepMeasurement3{initMeasurement3.timepoint()+dt,-70.,-70.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> stepMeasurement4{initMeasurement1.timepoint()+dt,220.,220.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> stepMeasurement5{initMeasurement2.timepoint()+dt,-220.,-220.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,};


    track1.initialization<Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>>
            (initMeasurement1);
    controlTrack1.initialization<Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>>
            (initMeasurement1);

    track2.initialization<Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>>
            (initMeasurement2);
    controlTrack2.initialization<Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>>
            (initMeasurement2);

    track3.initialization<Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>>
            (initMeasurement3);
    controlTrack3.initialization<Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>>
            (initMeasurement3);

    ASSERT_TRUE(track1.isInit()==true);
    ASSERT_TRUE(track2.isInit()==true);
    ASSERT_TRUE(track3.isInit()==true);
    ASSERT_TRUE(controlTrack1.isInit()==true);
    ASSERT_TRUE(controlTrack2.isInit()==true);
    ASSERT_TRUE(controlTrack3.isInit()==true);

    //----------------------------------------------------------------------
    auto controlTrack1_res = controlTrack1.step(stepMeasurement1);
    Eigen::MatrixXd controlState1 = controlTrack1_res.first;
    Eigen::MatrixXd controlCovariance1 = controlTrack1_res.second;

    auto controlTrack2_res = controlTrack2.step(stepMeasurement2);
    Eigen::MatrixXd controlState2 = controlTrack2_res.first;
    Eigen::MatrixXd controlCovariance2 = controlTrack2_res.second;

    auto controlTrack3_res = controlTrack3.step(stepMeasurement3);
    Eigen::MatrixXd controlState3 = controlTrack3_res.first;
    Eigen::MatrixXd controlCovariance3 = controlTrack3_res.second;
    //----------------------------------------------------------------------
    std::vector<Tracker::Track<Eigen::MatrixXd,Tracker::Measurement2<Eigen::MatrixXd>,Tracker::EstimatorInitializator4<Eigen::MatrixXd,Estimator::KF<Eigen::MatrixXd,stateModel,measureModel,noiseTransitionModel>,Tracker::Measurement2<Eigen::MatrixXd>>>*> tracks;
    tracks.push_back(&track1);
    tracks.push_back(&track2);
    tracks.push_back(&track3);

    std::vector<Tracker::Measurement2<Eigen::MatrixXd>> measurements;
    measurements.push_back(stepMeasurement1);
    measurements.push_back(stepMeasurement2);
    measurements.push_back(stepMeasurement3);
    measurements.push_back(stepMeasurement4);
    measurements.push_back(stepMeasurement5);
    //----------------------------------------------------------------------
    Eigen::MatrixXd association_matrix = Association::GNN_prototype_2<Eigen::MatrixXd,
                                                                      Tracker::Track<Eigen::MatrixXd,
                                                                                     Tracker::Measurement2<Eigen::MatrixXd>,
                                                                                     Tracker::EstimatorInitializator4<Eigen::MatrixXd,
                                                                                                                      Estimator::KF<Eigen::MatrixXd,
                                                                                                                                    stateModel,
                                                                                                                                    measureModel,
                                                                                                                                    noiseTransitionModel>,
                                                                                     Tracker::Measurement2<Eigen::MatrixXd>>>,
                                                                                     Tracker::Measurement2<Eigen::MatrixXd>>()(tracks, measurements);
    std::vector<std::pair<int,int>> result_pairs = Association::Auction_prototype<Eigen::MatrixXd>()(association_matrix/*,vec_prices*/);
    //----------------------------------------------------------------------

    for(int i=0;i<result_pairs.size();i++)
        tracks.at(result_pairs.at(i).first)->step(measurements.at(result_pairs.at(i).second));

    //----------------------------------------------------------------------
    ASSERT_TRUE(controlState1.isApprox(tracks.at(0)->getState(),0.00001));
    ASSERT_TRUE(controlCovariance1.isApprox(tracks.at(0)->getCovariance(),0.00001));
    ASSERT_TRUE(controlState2.isApprox(tracks.at(1)->getState(),0.00001));
    ASSERT_TRUE(controlCovariance2.isApprox(tracks.at(1)->getCovariance(),0.00001));
    ASSERT_TRUE(controlState3.isApprox(tracks.at(2)->getState(),0.00001));
    ASSERT_TRUE(controlCovariance3.isApprox(tracks.at(2)->getCovariance(),0.00001));
}
