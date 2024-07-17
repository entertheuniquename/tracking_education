#include "../source/kf.h"
#include "../source/ekf.h"
#include "../source/imm_prototype.h"
#include "../source/imm.h"
#include "../source/models.h"
#include <gtest/gtest.h>

TEST (IMM,imm_base_test) {
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

    Eigen::MatrixXd Q01(2,2);
    Q01 << 1., 0.,
           0., 1.;
    Eigen::MatrixXd Q02(2,2);
    Q02 << 0., 0.,
           0., 0.;
    Eigen::MatrixXd Q03(2,2);
    Q03 << 10., 0.,
           0., 10.;
    Eigen::MatrixXd z(2,1);
    z << 10., 20.;
    double t = 0.2;
    //----------------------------------------------------------------------

    Estimator::KF<Eigen::MatrixXd,
                  stateModel,
                  measureModel,
                  noiseTransitionModel>* kf1 = new Estimator::KF<Eigen::MatrixXd,
                                                                 stateModel,
                                                                 measureModel,
                                                                 noiseTransitionModel>(x0,P0,Q01,R);
    Estimator::KF<Eigen::MatrixXd,
                  stateModel,
                  measureModel,
                  noiseTransitionModel>* kf2 = new Estimator::KF<Eigen::MatrixXd,
                                                                 stateModel,
                                                                 measureModel,
                                                                 noiseTransitionModel>(x0,P0,Q02,R);
    Estimator::KF<Eigen::MatrixXd,
                  stateModel,
                  measureModel,
                  noiseTransitionModel>* kf3 = new Estimator::KF<Eigen::MatrixXd,
                                                                 stateModel,
                                                                 measureModel,
                                                                 noiseTransitionModel>(x0,P0,Q03,R);
    Estimator::KF<Eigen::MatrixXd,
                  stateModel,
                  measureModel,
                  noiseTransitionModel> kf01(x0,P0,Q01,R);
    Estimator::KF<Eigen::MatrixXd,
                  stateModel,
                  measureModel,
                  noiseTransitionModel> kf02(x0,P0,Q02,R);
    Estimator::KF<Eigen::MatrixXd,
                  stateModel,
                  measureModel,
                  noiseTransitionModel> kf03(x0,P0,Q03,R);

    Eigen::MatrixXd mu(1,3);
    mu << 0.3333, 0.3333, 0.3333;
    Eigen::MatrixXd trans(3,3);
    trans << 0.95, 0.025, 0.025,
             0.025, 0.95, 0.025,
             0.025, 0.025, 0.95;


//    Estimator::IMM_prototype<Eigen::MatrixXd,
//                   Estimator::KF<Eigen::MatrixXd,
//                                 stateModel,
//                                 measureModel,
//                                 noiseTransitionModel>,
//                   Estimator::KF<Eigen::MatrixXd,
//                                 stateModel,
//                                 measureModel,
//                                 noiseTransitionModel>,
//                   Estimator::KF<Eigen::MatrixXd,
//                                 stateModel,
//                                 measureModel,
//                                 noiseTransitionModel>> imm(kf1,kf2,kf3,mu,trans);
//    auto pred = imm.predict(t);
//    auto corr = imm.correct(z);

//    Eigen::MatrixXd xp1 = pred.first;
//    Eigen::MatrixXd Pp1 = pred.second;
//    Eigen::MatrixXd xc1 = corr.first;
//    Eigen::MatrixXd Pc1 = corr.second;
//    Eigen::MatrixXd mu1 = imm.getMU();

    Estimator::IMM<Eigen::MatrixXd,
                   Estimator::KF<Eigen::MatrixXd,
                                 stateModel,
                                 measureModel,
                                 noiseTransitionModel>,
                   Estimator::KF<Eigen::MatrixXd,
                                 stateModel,
                                 measureModel,
                                 noiseTransitionModel>,
                   Estimator::KF<Eigen::MatrixXd,
                                 stateModel,
                                 measureModel,
                                 noiseTransitionModel>> imm(mu,trans,kf01,kf02,kf03);

    auto pred = imm.predict(t);
    auto corr = imm.correct(z);

    Eigen::MatrixXd xp1 = pred.first;
    Eigen::MatrixXd Pp1 = pred.second;
    Eigen::MatrixXd xc1 = corr.first;
    Eigen::MatrixXd Pc1 = corr.second;
    Eigen::MatrixXd mu1 = imm.getMU();

    Estimator::IMM<Eigen::MatrixXd,
                   Estimator::KF<Eigen::MatrixXd,
                                 stateModel,
                                 measureModel,
                                 noiseTransitionModel>,
                   Estimator::KF<Eigen::MatrixXd,
                                 stateModel,
                                 measureModel,
                                 noiseTransitionModel>,
                   Estimator::KF<Eigen::MatrixXd,
                                 stateModel,
                                 measureModel,
                                 noiseTransitionModel>> imm2(mu,trans,x0,P0,Q01,R);

    auto pred2 = imm2.predict(t);
    auto corr2 = imm2.correct(z);

    Eigen::MatrixXd xp2 = pred2.first;
    Eigen::MatrixXd Pp2 = pred2.second;
    Eigen::MatrixXd xc2 = corr2.first;
    Eigen::MatrixXd Pc2 = corr2.second;
    Eigen::MatrixXd mu2 = imm2.getMU();

    //FROM filterpy_test_imm.py
    Eigen::MatrixXd filterpy_xp1(4,1);
    filterpy_xp1 << 1.4, 2., 3.8, 4.;
    Eigen::MatrixXd filterpy_Pp1(4,4);
    filterpy_Pp1 << 1.04146667, 0.214666667, 0., 0.,
                    0.214666667, 1.14666667, 0., 0.,
                    0., 0., 1.04146667, 0.214666667,
                    0., 0., 0.214666667, 1.14666667;
    Eigen::MatrixXd filterpy_xc1(4,1);
    filterpy_xc1 << 5.78761117, 2.90974556, 12.06503499, 5.71370675;
    Eigen::MatrixXd filterpy_Pc1(4,4);
    filterpy_Pc1 << 0.510201720, 0.106074753, 0.0000270794591, 0.000547005074,
                    0.106074753, 1.14271000, 0.000547005074, 0.0110495025,
                    0.0000270794591, 0.000547005074, 0.510238355, 0.106814772,
                    0.000547005074, 0.0110495025, 0.106814772, 1.15765840;

    Eigen::MatrixXd filterpy_mu1(1,3);
    filterpy_mu1 << 0.3186357, 0.31358825, 0.36777604;

    Eigen::MatrixXd filterpy_likelihood1(1,3);
    filterpy_likelihood1 << 1.23344221e-37, 1.21390347e-37, 1.42366499e-37;

    Eigen::MatrixXd filterpy_Se1(2,2);
    filterpy_Se1 << 2.0404, 1.38050658e-32,
                    1.38050658e-32, 2.0404;

    Eigen::MatrixXd filterpy_Se2(2,2);
    filterpy_Se2 << 2.04, 1.38050658e-32,
                    1.38050658e-32, 2.04;

    Eigen::MatrixXd filterpy_Se3(2,2);
    filterpy_Se3 << 2.044, 0.,
                    0., 2.044;

    Eigen::MatrixXd filterpy_cbar1(1,3);
    filterpy_cbar1 << 0.31973803, 0.31506913, 0.36519284;

    Eigen::MatrixXd filterpy_omega1(3,3);
    filterpy_omega1 << 0.9467248, 0.025283, 0.02181284,
                       0.02451916, 0.94553483, 0.02146731,
                       0.02875605, 0.02918217, 0.95671986;

    ASSERT_TRUE(xp1.isApprox(filterpy_xp1,0.001));
    ASSERT_TRUE(Pp1.isApprox(filterpy_Pp1,0.001));
    ASSERT_TRUE(xc1.isApprox(filterpy_xc1,0.001));
    ASSERT_TRUE(Pc1.isApprox(filterpy_Pc1,0.001));
    ASSERT_TRUE(mu1.isApprox(filterpy_mu1,0.001));
}
