#include "../source/track.h"
#include "../source/kf.h"
#include "../source/ekf.h"
#include "../source/imm.h"
#include "../source/models.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>

TEST (Track,track_base_test) {
    Tracker::Track<Eigen::MatrixXd,Tracker::Measurement3<Eigen::MatrixXd>> track10;
    ASSERT_TRUE(track10.isInit()==false);

    Tracker::Measurement3<Eigen::MatrixXd> m10{12345.,100.,200.,300.,40.,50.,60.,7.,8.,9.,1.1,1.,1.,1.,0.,0.,0.};

    track10.initialization<Tracker::EstimatorInitializator10<Eigen::MatrixXd,
                                                             Estimator::KF<Eigen::MatrixXd,
                                                                           Models10::FCV<Eigen::MatrixXd>,
                                                                           Models10::H<Eigen::MatrixXd>,
                                                                           Models10::G<Eigen::MatrixXd>>,
                                                             Tracker::Measurement3<Eigen::MatrixXd>>>(m10);
    ASSERT_TRUE(track10.isInit()==true);
    track10.initialization<Tracker::EstimatorInitializator10<Eigen::MatrixXd,
                                                             Estimator::EKF<Eigen::MatrixXd,
                                                                            Models10::FCA<Eigen::MatrixXd>,
                                                                            Models10::H<Eigen::MatrixXd>,
                                                                            Models10::G<Eigen::MatrixXd>,
                                                                            Models10::FCA_Jacobian<Eigen::MatrixXd>,
                                                                            Models10::H_Jacobian<Eigen::MatrixXd>>,
                                                             Tracker::Measurement3<Eigen::MatrixXd>>>(m10);
    ASSERT_TRUE(track10.isInit()==true);
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
    //----------------------------------------------------------------------
    Tracker::Track<Eigen::MatrixXd,Tracker::Measurement2<Eigen::MatrixXd>> track4;
    ASSERT_TRUE(track4.isInit()==false);

    Tracker::Measurement2<Eigen::MatrixXd> m4_init{0.,1.,3.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,};
    Tracker::Measurement2<Eigen::MatrixXd> m4_step{m4_init.timepoint()+0.2,10.,20.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,};

    track4.initialization<Tracker::EstimatorInitializator4<Eigen::MatrixXd,
                                                           Estimator::KF<Eigen::MatrixXd,
                                                                         stateModel,
                                                                         measureModel,
                                                                         noiseTransitionModel>,
                                                           Tracker::Measurement2<Eigen::MatrixXd>>>(m4_init);
    ASSERT_TRUE(track4.isInit()==true);

    auto track4step = track4.step(m4_step);

    Eigen::MatrixXd xc1 = track4step.first;
    Eigen::MatrixXd Pc1 = track4step.second;

    //FROM filterpy_test_track.py
    Eigen::MatrixXd filterpy_xp1(4,1);
    filterpy_xp1 << 1., 0., 3., 0.;
    Eigen::MatrixXd filterpy_Pp1(4,4);
    filterpy_Pp1 << 1.0004, 0.004,  0.    , 0.   ,
                    0.004 , 0.04 ,  0.    , 0.   ,
                    0.    , 0.   ,  1.0004, 0.004,
                    0.    , 0.   ,  0.004 , 0.04 ;
    Eigen::MatrixXd filterpy_xc1(4,1);
    filterpy_xc1 << 5.50089982, 0.0179964, 11.50169966, 0.0339932;
    Eigen::MatrixXd filterpy_Pc1(4,4);
    filterpy_Pc1 << 0.50009998, 0.0019996, 0., 0.,
                    0.0019996, 0.039992, 0., 0.,
                    0., 0., 0.50009998, 0.0019996,
                    0., 0., 0.0019996, 0.039992;

    ASSERT_TRUE(xc1.isApprox(filterpy_xc1,0.001));
    ASSERT_TRUE(Pc1.isApprox(filterpy_Pc1,0.001));

}
