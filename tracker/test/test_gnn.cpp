#include "../source/kf.h"
#include "../source/gnn.h"
#include <gtest/gtest.h>

TEST (GlobalNearestNeighbor,GlobalNearestNeighbor_base_test) {
    //----------------------------------------------------------------------
    ASSERT_TRUE(true);
    //----------------------------------------------------------------------
    struct stateModel
    {
        Eigen::MatrixXd operator()(const Eigen::MatrixXd& x, double T, MatrixXd state=MatrixXd{})
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
    using MeasurementType = Tracker::Measurement2<MatrixType>;
    using EstimatorType = Estimator::KF<MatrixType,stateModel,measureModel,noiseTransitionModel>;
    using EstimatorInitializatorType = Tracker::EstimatorInitializator4<MatrixType,EstimatorType,MeasurementType>;
    using TrackType = Tracker::Track<MatrixType,MeasurementType,EstimatorInitializatorType>;
    //----------------------------------------------------------------------
    TrackType track1;
    TrackType controlTrack1;

    TrackType track2;
    TrackType controlTrack2;

    TrackType track3;
    TrackType controlTrack3;

    ASSERT_TRUE(track1.isInit()==false);
    ASSERT_TRUE(track2.isInit()==false);
    ASSERT_TRUE(track3.isInit()==false);
    ASSERT_TRUE(controlTrack1.isInit()==false);
    ASSERT_TRUE(controlTrack2.isInit()==false);
    ASSERT_TRUE(controlTrack3.isInit()==false);

    MeasurementType initMeasurement1{0,0.,100.,100.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    MeasurementType initMeasurement2{0,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    MeasurementType initMeasurement3{0,0.,-100.,-100.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};

    MeasurementType stepMeasurement1{0,initMeasurement1.timepoint()+dt,110.,110.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,};
    MeasurementType stepMeasurement2{0,initMeasurement2.timepoint()+dt,20.,20.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,};
    MeasurementType stepMeasurement3{0,initMeasurement3.timepoint()+dt,-70.,-70.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,};
    MeasurementType stepMeasurement4{0,initMeasurement1.timepoint()+dt,220.,220.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,};
    MeasurementType stepMeasurement5{0,initMeasurement2.timepoint()+dt,-220.,-220.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,};


    track1.initialization<EstimatorInitializatorType>(initMeasurement1);
    controlTrack1.initialization<EstimatorInitializatorType>(initMeasurement1);

    track2.initialization<EstimatorInitializatorType>(initMeasurement2);
    controlTrack2.initialization<EstimatorInitializatorType>(initMeasurement2);

    track3.initialization<EstimatorInitializatorType>(initMeasurement3);
    controlTrack3.initialization<EstimatorInitializatorType>(initMeasurement3);

    ASSERT_TRUE(track1.isInit()==true);
    ASSERT_TRUE(track2.isInit()==true);
    ASSERT_TRUE(track3.isInit()==true);
    ASSERT_TRUE(controlTrack1.isInit()==true);
    ASSERT_TRUE(controlTrack2.isInit()==true);
    ASSERT_TRUE(controlTrack3.isInit()==true);
    //----------------------------------------------------------------------
    auto controlTrack1_res = controlTrack1.step(stepMeasurement1);
    MatrixType controlState1 = controlTrack1_res.first;
    MatrixType controlCovariance1 = controlTrack1_res.second;

    auto controlTrack2_res = controlTrack2.step(stepMeasurement2);
    MatrixType controlState2 = controlTrack2_res.first;
    MatrixType controlCovariance2 = controlTrack2_res.second;

    auto controlTrack3_res = controlTrack3.step(stepMeasurement3);
    MatrixType controlState3 = controlTrack3_res.first;
    MatrixType controlCovariance3 = controlTrack3_res.second;
    //----------------------------------------------------------------------
    std::vector<TrackType*> po_tracks;
    po_tracks.push_back(&track1);
    po_tracks.push_back(&track2);
    po_tracks.push_back(&track3);

    std::vector<MeasurementType> measurements;
    measurements.push_back(stepMeasurement1);
    measurements.push_back(stepMeasurement2);
    measurements.push_back(stepMeasurement3);
    measurements.push_back(stepMeasurement4);
    measurements.push_back(stepMeasurement5);
    //----------------------------------------------------------------------
    MatrixType am = Association::GlobalNearestNeighbor<MatrixType,TrackType,MeasurementType>()(po_tracks,measurements);
    //std::cout << "am:" << std::endl << am << std::endl;
    std::map<int,int> res_map = Association::Auction<MatrixType>()(am);
    //std::cout << "res_map: ";for(auto i : res_map)std::cout << "[" << i.first << "]+(" << i.second << ") ";std::cout << std::endl;
    //----------------------------------------------------------------------
    for(auto i : res_map)
        po_tracks[i.first]->step(measurements[i.second]);
    //----------------------------------------------------------------------
    for(auto i : res_map)
        ASSERT_TRUE(i.first==i.second);

    ASSERT_TRUE(controlState1.isApprox(track1.getState(),0.00001));
    ASSERT_TRUE(controlCovariance1.isApprox(track1.getCovariance(),0.00001));
    ASSERT_TRUE(controlState2.isApprox(track2.getState(),0.00001));
    ASSERT_TRUE(controlCovariance2.isApprox(track2.getCovariance(),0.00001));
    ASSERT_TRUE(controlState3.isApprox(track3.getState(),0.00001));
    ASSERT_TRUE(controlCovariance3.isApprox(track3.getCovariance(),0.00001));
    //----------------------------------------------------------------------
}

TEST (GlobalNearestNeighbor,AuctionMinimization_base_test) {
    //----------------------------------------------------------------------
    ASSERT_TRUE(true);
    //----------------------------------------------------------------------
    using M = Eigen::MatrixXd;

    M in_matrix(3,5);
    in_matrix << 58.75465191359883, 54.83914821643492, 25.32973896426497, 94.65996701616142, 39.9317056205971,
                 2.084167616451886, 55.801289123226205, 97.74098167832638, 23.870226157503414, 70.63699757466911,
                 68.55108984107467, 56.91822306774177, 45.02701785835559, 36.945122369903466, 93.05270595545396;

    double max_value = 99.9;

    Association::AuctionMinimization<M> auc_min;
    auto res = auc_min(in_matrix,max_value);

    double expectet_sum = 64.35902895062031;
    std::map<int,int> expected_res;
    expected_res[0]=2;
    expected_res[1]=0;
    expected_res[2]=3;

    ASSERT_TRUE(expectet_sum==res.first);
    ASSERT_TRUE(expected_res==res.second);

    //----------------------------------------------------------------------

    M in_matrix2(5,2);
    in_matrix2 << 77.82091753181122, 93.19021127325864,
                 4.721444450620989, 30.559511641847347,
                 83.61581176957598, 15.055935660185419,
                 97.87017504698017, 77.22239858079969,
                 30.32056781418549, 62.59627179028928;

    double max_value2 = 99.9;

    Association::AuctionMinimization<M> auc_min2;
    auto res2 = auc_min2(in_matrix2,max_value2);

    double expectet_sum2 = 19.77738011080641;
    std::map<int,int> expected_res2;
    expected_res2[1]=0;
    expected_res2[2]=1;

    ASSERT_TRUE(expectet_sum2==res2.first);
    ASSERT_TRUE(expected_res2==res2.second);
}
