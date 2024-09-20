#include "../source/track.h"
#include "../source/kf.h"
#include "../source/ekf.h"
#include "../source/imm.h"
#include "../source/models.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>

TEST (Track,track_base_test) {
    //----------------------------------------------------------------------
    using MatrixType_10 = Eigen::MatrixXd;
    using MeasurementType_10 = Tracker::Measurement3<MatrixType_10>;
    using FCVModelType_10 = Models10::FCV<MatrixType_10>;
    using HModelType_10 = Models10::H<MatrixType_10>;
    using GModelType_10 = Models10::G<MatrixType_10>;
    using EstimatorType_10 = Estimator::KF<MatrixType_10,FCVModelType_10,HModelType_10,GModelType_10>;
    using EstimatorInitializatorType_10 = Tracker::EstimatorInitializator10<MatrixType_10,EstimatorType_10,MeasurementType_10>;
    using TrackType_10 = Tracker::Track<MatrixType_10,MeasurementType_10,EstimatorInitializatorType_10>;
    //----------------------------------------------------------------------
    TrackType_10 track10;
    ASSERT_TRUE(track10.isInit()==false);

    MeasurementType_10 m10{0,12345.,100.,200.,300.,40.,50.,60.,7.,8.,9.,1.1,1.,1.,1.,0.,0.,0.};

    track10.initialization(m10);
    ASSERT_TRUE(track10.isInit()==true);

    track10.initialization<EstimatorInitializatorType_10>(m10);
    ASSERT_TRUE(track10.isInit()==true);
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
    //----------------------------------------------------------------------
    using MatrixType_4 = Eigen::MatrixXd;
    using MeasurementType_4 = Tracker::Measurement2<MatrixType_4>;
    using FCVModelType_4 = stateModel;
    using HModelType_4 = measureModel;
    using GModelType_4 = noiseTransitionModel;
    using EstimatorType_4 = Estimator::KF<MatrixType_4,FCVModelType_4,HModelType_4,GModelType_4>;
    using EstimatorInitializatorType_4 = Tracker::EstimatorInitializator4<MatrixType_4,EstimatorType_4,MeasurementType_4>;
    using TrackType_4 = Tracker::Track<MatrixType_4,MeasurementType_4,EstimatorInitializatorType_4>;
    //----------------------------------------------------------------------
    TrackType_4 track4;
    ASSERT_TRUE(track4.isInit()==false);

    MeasurementType_4 m4_init{0,0.,1.,3.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    MeasurementType_4 m4_step{0,m4_init.timepoint()+0.2,10.,20.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,};

    track4.initialization(m4_init);
    ASSERT_TRUE(track4.isInit()==true);

    auto track4step = track4.step(m4_step);

    MatrixType_4 xc1 = track4step.first;
    MatrixType_4 Pc1 = track4step.second;

    //FROM filterpy_test_track.py
    MatrixType_4 filterpy_xp1(4,1);
    filterpy_xp1 << 1., 0., 3., 0.;
    MatrixType_4 filterpy_Pp1(4,4);
    filterpy_Pp1 << 1.0004, 0.004,  0.    , 0.   ,
                    0.004 , 0.04 ,  0.    , 0.   ,
                    0.    , 0.   ,  1.0004, 0.004,
                    0.    , 0.   ,  0.004 , 0.04 ;
    MatrixType_4 filterpy_xc1(4,1);
    filterpy_xc1 << 5.50089982, 0.0179964, 11.50169966, 0.0339932;
    MatrixType_4 filterpy_Pc1(4,4);
    filterpy_Pc1 << 0.50009998, 0.0019996, 0., 0.,
                    0.0019996, 0.039992, 0., 0.,
                    0., 0., 0.50009998, 0.0019996,
                    0., 0., 0.0019996, 0.039992;

    ASSERT_TRUE(xc1.isApprox(filterpy_xc1,0.001));
    ASSERT_TRUE(Pc1.isApprox(filterpy_Pc1,0.001));

}

//TEST (Track,track_getMeasurementPredictData_test) {
//    //----------------------------------------------------------------------
//    using MatrixType_10 = Eigen::MatrixXd;
//    using MeasurementType_10 = Tracker::Measurement3<MatrixType_10>;
//    using FCVModelType_10 = Models10::FCV<MatrixType_10>;
//    using HModelType_10 = Models10::H<MatrixType_10>;
//    using GModelType_10 = Models10::G<MatrixType_10>;
//    using EstimatorType_10 = Estimator::KF<MatrixType_10,FCVModelType_10,HModelType_10,GModelType_10>;
//    using EstimatorInitializatorType_10 = Tracker::EstimatorInitializator10<MatrixType_10,EstimatorType_10,MeasurementType_10>;
//    using TrackType_10 = Tracker::Track<MatrixType_10,MeasurementType_10,EstimatorInitializatorType_10>;
//    //----------------------------------------------------------------------
//    TrackType_10 track10;
//    ASSERT_TRUE(track10.isInit()==false);

//    MeasurementType_10 m10{12345.,100.,200.,300.,40.,50.,60.,7.,8.,9.,1.1,1.,1.,1.,0.,0.,0.};

//    track10.initialization(m10);
//    ASSERT_TRUE(track10.isInit()==true);

//    MeasurementType_10 m10_step1{12345.+6.,110.,210.,310.,41.,51.,61.,0.,0.,0.,0.0,0.,0.,0.,0.,0.,0.};
//    auto data0 = track10.getMeasurementPredictData(m10_step1.timepoint()-m10.timepoint());
//    MatrixType_10 zp0 = data0.first;
//    MatrixType_10 Se0 = data0.second;
//    auto data_step = track10.step(m10_step1);
//    MatrixType_10 zp1 = track10.getMeasurementPredict();
//    MatrixType_10 Se1 = track10.getCovarianceOfMeasurementPredict();

//    //compare
//    //std::cout << "*********************************************" << std::endl;
//    //std::cout << "Se0" << std::endl << Se0 << std::endl;
//    //std::cout << "Se1" << std::endl << Se1 << std::endl;
//    //std::cout << "*********************************************" << std::endl;

//    ASSERT_TRUE(zp0.isApprox(zp1,0.00001));
//    ASSERT_TRUE(Se0.isApprox(Se1,0.00001));

//}

//TEST (Track,track_imm_test) {
//    //----------------------------------------------------------------------
//    using MatrixType_10 = Eigen::MatrixXd;
//    using MeasurementType_10 = Tracker::Measurement3<MatrixType_10>;
//    using FCVModelType_10 = Models10::FCV<MatrixType_10>;
//    using HModelType_10 = Models10::H<MatrixType_10>;
//    using GModelType_10 = Models10::G<MatrixType_10>;
//    using EstimatorKFType_10 = Estimator::KF<MatrixType_10,FCVModelType_10,HModelType_10,GModelType_10>;
//    using EstimatorIMMType_10 = Estimator::IMM<MatrixType_10,EstimatorKFType_10,EstimatorKFType_10,EstimatorKFType_10>;
//    using EstimatorInitializatorIMMType_10 = Tracker::EstimatorInitializator10<MatrixType_10,EstimatorIMMType_10,MeasurementType_10,MatrixType_10,MatrixType_10>;
//    using EstimatorInitializatorIMMType_10_2 = Tracker::EstimatorInitializator10_IMM3<MatrixType_10,EstimatorIMMType_10,MeasurementType_10>;
//    using TrackType_10 = Tracker::Track<MatrixType_10,MeasurementType_10,EstimatorInitializatorIMMType_10>;
//    using TrackType_10_2 = Tracker::Track<MatrixType_10,MeasurementType_10,EstimatorInitializatorIMMType_10_2>;
//    //----------------------------------------------------------------------
//    TrackType_10 track10;
//    ASSERT_TRUE(track10.isInit()==false);

//    TrackType_10_2 track10_2;
//    ASSERT_TRUE(track10_2.isInit()==false);

//    MeasurementType_10 m10{12345.,100.,200.,300.,40.,50.,60.,7.,8.,9.,1.1,1.,1.,1.,0.,0.,0.};

//    Eigen::MatrixXd mu(1,3);
//    mu << 0.3333, 0.3333, 0.3333;
//    Eigen::MatrixXd trans(3,3);
//    trans << 0.95, 0.025, 0.025,
//             0.025, 0.95, 0.025,
//             0.025, 0.025, 0.95;

//    track10.initialization(m10,mu,trans);
//    ASSERT_TRUE(track10.isInit()==true);

//    track10_2.initialization(m10);
//    ASSERT_TRUE(track10_2.isInit()==true);

//    MeasurementType_10 m10_step1{12345.+6.,110.,210.,310.,41.,51.,61.,0.,0.,0.,0.0,0.,0.,0.,0.,0.,0.};
//    auto data0 = track10.getMeasurementPredictData(m10_step1.timepoint()-m10.timepoint());
//    MatrixType_10 zp0 = data0.first;
//    MatrixType_10 Se0 = data0.second;
//    auto data_step = track10.step(m10_step1);
//    MatrixType_10 zp1 = track10.getMeasurementPredict();
//    MatrixType_10 Se1 = track10.getCovarianceOfMeasurementPredict();
//    auto data_step_2 = track10_2.step(m10_step1);
//    MatrixType_10 zp1_2 = track10_2.getMeasurementPredict();
//    MatrixType_10 Se1_2 = track10_2.getCovarianceOfMeasurementPredict();

//    ASSERT_TRUE(zp0.isApprox(zp1,0.00001));
//    ASSERT_TRUE(Se0.isApprox(Se1,0.00001));

//    ASSERT_TRUE(zp0.isApprox(zp1_2,0.00001));
//    ASSERT_TRUE(Se0.isApprox(Se1_2,0.00001));

//}
