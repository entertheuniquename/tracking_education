#include "../source/kf.h"
#include "../source/models.h"
#include <gtest/gtest.h>

TEST (KF,kf_base_test) {
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
    Eigen::MatrixXd x0(4,1);
    x0 << 1., 2., 3., 4.;
    Eigen::MatrixXd R(2,2);
    R << 1., 0.,
         0., 1.;
    Eigen::MatrixXd P0(4,4);
    P0 << 1., 0., 0., 0.,
          0., 1., 0., 0.,
          0., 0., 1., 0.,
          0., 0., 0., 1.;

    Eigen::MatrixXd Q0(2,2);
    Q0 << 1., 0.,
          0., 1.;
    Eigen::MatrixXd z(2,1);
    z << 10., 20.;
    double t = 0.2;
    //----------------------------------------------------------------------

    Estimator::KFMath<Eigen::MatrixXd> kf_math;
    Estimator::KF<Eigen::MatrixXd,
                  stateModel,
                  measureModel,
                  noiseTransitionModel> kf(x0,P0,Q0,R);

    auto pred = kf.predict(t);
    Eigen::MatrixXd xp1 = pred.first;
    Eigen::MatrixXd Pp1 = pred.second;

    auto corr = kf.correct(z);
    Eigen::MatrixXd xc1 = corr.first;
    Eigen::MatrixXd Pc1 = corr.second;
    Eigen::MatrixXd Sc1 = kf.getCovarianceOfMeasurementPredict();

    //FROM filterpy_test_kf.py
    Eigen::MatrixXd filterpy_xp1(4,1);
    filterpy_xp1 << 1.4, 2., 3.8, 4.;
    Eigen::MatrixXd filterpy_Pp1(4,4);
    filterpy_Pp1 << 1.0404, 0.204,  0.    , 0.   ,
                    0.204 , 1.04 ,  0.    , 0.   ,
                    0.    , 0.   ,  1.0404, 0.204,
                    0.    , 0.   ,  0.204 , 1.04 ;
    Eigen::MatrixXd filterpy_xc1(4,1);
    filterpy_xc1 << 5.78514017, 2.85983141, 12.06038032, 5.61968242;
    Eigen::MatrixXd filterpy_Pc1(4,4);
    filterpy_Pc1 << 0.50990002, 0.0999804,  0.        , 0.       ,
                    0.0999804 , 1.019604 ,  0.        , 0.       ,
                    0.        , 0.       ,  0.50990002, 0.0999804,
                    0.        , 0.       ,  0.0999804 , 1.019604 ;
    Eigen::MatrixXd filterpy_Sc1(2,2);
    filterpy_Sc1 << 2.0404, 0.,
                    0., 2.0404;

    ASSERT_TRUE(xp1.isApprox(filterpy_xp1,0.001));
    ASSERT_TRUE(Pp1.isApprox(filterpy_Pp1,0.001));
    ASSERT_TRUE(xc1.isApprox(filterpy_xc1,0.001));
    ASSERT_TRUE(Pc1.isApprox(filterpy_Pc1,0.001));
    ASSERT_TRUE(Sc1.isApprox(filterpy_Sc1,0.001));
}
