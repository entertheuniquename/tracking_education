#include "../source/ekf.h"
#include "../source/models.h"
#include <gtest/gtest.h>

TEST (EKF,ekf_base_test) {
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
    struct stateModelJacobian
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
    struct measureModelPol
    {
        Eigen::MatrixXd operator()(const Eigen::MatrixXd& x, Eigen::MatrixXd z = Eigen::MatrixXd{}, Eigen::MatrixXd state = Eigen::MatrixXd{})
        {
            double X = x(0);
            double Y = x(2);
            double angle = atan2(Y,X);
            double range = sqrt(X*X+Y*Y);
            Eigen::MatrixXd r(1,2);
            r << range, angle;
            //std::cout << "measureModelPol:" << std::endl << r.transpose() << std::endl;
            return r.transpose();
        }
    };
    struct measureModelJacobianPol
    {
        Eigen::MatrixXd matrix(Eigen::MatrixXd state = Eigen::MatrixXd{})
        {
            double X = state(0);
            double Y = state(2);
            Eigen::MatrixXd J(2,4);
            J(0,0) = X/std::sqrt((X*X)+(Y*Y));
            J(0,1) = 0.;
            J(0,2) = Y/std::sqrt((X*X)+(Y*Y));
            J(0,3) = 0.;

            J(1,0) = -Y/((X*X)*(1.+((Y*Y)/(X*X))));
            J(1,1) = 0.;
            J(1,2) = 1./(X*(1.+((Y*Y)/(X*X))));
            J(1,3) = 0.;
            //std::cout << "measureModelJacobianPol:" << std::endl << J*x << std::endl;
            return J;
        }
        Eigen::MatrixXd operator()(const Eigen::MatrixXd& x, Eigen::MatrixXd z = Eigen::MatrixXd{}, Eigen::MatrixXd state = Eigen::MatrixXd{})
        {return matrix(state)*x;}
    };
    struct measureModelDec
    {
        Eigen::MatrixXd operator()(const Eigen::MatrixXd& x, Eigen::MatrixXd z = Eigen::MatrixXd{}, Eigen::MatrixXd state = Eigen::MatrixXd{})
        {
            Eigen::MatrixXd H(2,4);
            H << 1., 0., 0., 0.,
                 0., 0., 1., 0.;
            //std::cout << "measureModelDec:" << std::endl << H*x << std::endl;
            return H*x;
        }
    };
    struct measureModelJacobianDec
    {
        Eigen::MatrixXd operator()(const Eigen::MatrixXd& x, Eigen::MatrixXd z = Eigen::MatrixXd{}, Eigen::MatrixXd state = Eigen::MatrixXd{})
        {
            Eigen::MatrixXd H(2,4);
            H << 1., 0., 0., 0.,
                 0., 0., 1., 0.;
            //std::cout << "measureModelJacobianDec:" << std::endl << H*x << std::endl;
            return H*x;
        }
    };
    struct noiseTransitionModel
    {
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
    Eigen::MatrixXd x0(4,1);
    x0 << 20., 2., 20., 2.;
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
    Eigen::MatrixXd z1_dec(2,1);
    z1_dec << 25., 25.;
    Eigen::MatrixXd z1_pol(2,1);
    z1_pol << 35.355, /*45.*/0.785;
    Eigen::MatrixXd z2_dec(2,1);
    z2_dec << 30., 20.;
    Eigen::MatrixXd z2_pol(2,1);
    z2_pol << 36.056, /*33.69*/0.588;
    double t = 0.2;
    //----------------------------------------------------------------------
    //[CREATE]
    Estimator::EKF<Eigen::MatrixXd,
                   stateModel,
                   measureModelDec,
                   noiseTransitionModel,
                   stateModelJacobian,
                   measureModelJacobianDec> ekf_dec(x0,P0,Q0,R);

    Eigen::MatrixXd x_dec = ekf_dec.getState();
    Eigen::MatrixXd P_dec = ekf_dec.getCovariance();
    Eigen::MatrixXd Q_dec = ekf_dec.getGQG(t);
    Eigen::MatrixXd R_dec = ekf_dec.getMeasurementNoise();

    //std::cout << "xp1_dec:" << std::endl << xp1_dec << std::endl;
    //std::cout << "Pp1_dec:" << std::endl << Pp1_dec << std::endl;
    //std::cout << "xc1_dec:" << std::endl << xc1_dec << std::endl;
    //std::cout << "Q_dec:" << std::endl << Q_dec << std::endl;
    //std::cout << "R_dec:" << std::endl << R_dec << std::endl;
    //std::cout << "Pc1_dec:" << std::endl << Pc1_dec << std::endl;

    //FROM filterpy_test_ekf_pol.py
    Eigen::MatrixXd filterpy_x_dec(4,1);
    filterpy_x_dec << 20., 2., 20., 2.;
    Eigen::MatrixXd filterpy_P_dec(4,4);
    filterpy_P_dec << 1., 0., 0., 0.,
                      0., 1., 0., 0.,
                      0., 0., 1., 0.,
                      0., 0., 0., 1.;
    Eigen::MatrixXd filterpy_Q_dec(4,4);
    filterpy_Q_dec << 0.0004 ,0.004,      0.,    0.,
                       0.004 , 0.04,      0.,    0.,
                           0.,   0., 0.0004 , 0.004,
                           0.,   0.,  0.004 ,  0.04;
    Eigen::MatrixXd filterpy_R_dec(2,2);
    filterpy_R_dec << 1.,0.,
                      0.,1.;

    ASSERT_TRUE(x_dec.isApprox(filterpy_x_dec,0.001));
    ASSERT_TRUE(P_dec.isApprox(filterpy_P_dec,0.001));
    ASSERT_TRUE(Q_dec.isApprox(filterpy_Q_dec,0.001));
    ASSERT_TRUE(R_dec.isApprox(filterpy_R_dec,0.001));

    //[STEP-1]
    auto zS1_dec = ekf_dec.getMeasurementPredictData(t);
    Eigen::MatrixXd S1_dec = zS1_dec.second;

    auto pred_dec = ekf_dec.predict(t);
    Eigen::MatrixXd xp1_dec = pred_dec.first;
    Eigen::MatrixXd Pp1_dec = pred_dec.second;

    Eigen::MatrixXd filterpy_xp1_dec(4,1);
    filterpy_xp1_dec << 20.4, 2., 20.4, 2.;
    Eigen::MatrixXd filterpy_Pp1_dec(4,4);
    filterpy_Pp1_dec << 1.0404, 0.204,     0.,    0.,
                         0.204,  1.04,     0.,    0.,
                            0.,    0., 1.0404, 0.204,
                            0.,    0.,  0.204,  1.04;

    auto corr_dec = ekf_dec.correct(z1_dec);
    Eigen::MatrixXd xc1_dec = corr_dec.first;
    Eigen::MatrixXd Pc1_dec = corr_dec.second;

    Eigen::MatrixXd filterpy_xc1_dec(4,1);
    filterpy_xc1_dec << 22.74554009, 2.45990982, 22.74554009, 2.45990982;
    Eigen::MatrixXd filterpy_Pc1_dec(4,4);
    filterpy_Pc1_dec << 0.50990002, 0.0999804,         0.,        0.,
                         0.0999804,  1.019604,         0.,        0.,
                                0.,        0., 0.50990002, 0.0999804,
                                0.,        0.,  0.0999804,  1.019604;
    Eigen::MatrixXd filterpy_S1_dec(2,2);
    filterpy_S1_dec << 2.0404,      0,
                           0., 2.0404;
    Eigen::MatrixXd filterpy_K1_dec(2,4);
    filterpy_K1_dec << 0.50990002,         0.,
                        0.0999804,         0.,
                               0., 0.50990002,
                               0.,  0.0999804;
    Eigen::MatrixXd filterpy_z1_dec(2,1);
    filterpy_z1_dec << 25., 25.;
    ASSERT_TRUE(xp1_dec.isApprox(filterpy_xp1_dec,0.001));
    ASSERT_TRUE(Pp1_dec.isApprox(filterpy_Pp1_dec,0.001));
    ASSERT_TRUE(xc1_dec.isApprox(filterpy_xc1_dec,0.001));
    ASSERT_TRUE(Pc1_dec.isApprox(filterpy_Pc1_dec,0.001));
    ASSERT_TRUE(S1_dec.isApprox(filterpy_S1_dec,0.001));

    //[STEP-2]
    auto zS2_dec = ekf_dec.getMeasurementPredictData(t);
    Eigen::MatrixXd S2_dec = zS2_dec.second;

    auto pred_dec2 = ekf_dec.predict(t);
    Eigen::MatrixXd xp2_dec = pred_dec2.first;
    Eigen::MatrixXd Pp2_dec = pred_dec2.second;

    Eigen::MatrixXd filterpy_xp2_dec(4,1);
    filterpy_xp2_dec << 23.23752205, 2.45990982, 23.23752205, 2.45990982;
    Eigen::MatrixXd filterpy_Pp2_dec(4,4);
    filterpy_Pp2_dec << 0.59107634, 0.3079012,         0.,        0.,
                         0.3079012,  1.059604,         0.,        0.,
                                0.,        0., 0.59107634, 0.3079012,
                                0.,        0.,  0.3079012,  1.059604;

    auto corr_dec2 = ekf_dec.correct(z2_dec);
    Eigen::MatrixXd xc2_dec = corr_dec2.first;
    Eigen::MatrixXd Pc2_dec = corr_dec2.second;

    Eigen::MatrixXd filterpy_xc2_dec(4,1);
    filterpy_xc2_dec << 25.74974639, 3.76856799, 22.03479995, 1.83339248;
    Eigen::MatrixXd filterpy_Pc2_dec(4,4);
    filterpy_Pc2_dec << 0.37149464, 0.19351755, 0.        , 0.       ,
                        0.19351755, 1.00001971, 0.        , 0.       ,
                        0.        , 0.        , 0.37149464, 0.19351755,
                        0.        , 0.        , 0.19351755, 1.00001971;
    Eigen::MatrixXd filterpy_S2_dec(2,2);
    filterpy_S2_dec << 1.59107634, 0.        ,
                       0.        , 1.59107634;
    Eigen::MatrixXd filterpy_K2_dec(2,4);
    filterpy_K2_dec << 0.37149464, 0.        ,
                       0.19351755, 0.        ,
                       0.        , 0.37149464,
                       0.        , 0.19351755;
    Eigen::MatrixXd filterpy_z2_dec(2,1);
    filterpy_z2_dec << 30., 20.;
    ASSERT_TRUE(xp2_dec.isApprox(filterpy_xp2_dec,0.001));
    ASSERT_TRUE(Pp2_dec.isApprox(filterpy_Pp2_dec,0.001));
    ASSERT_TRUE(xc2_dec.isApprox(filterpy_xc2_dec,0.001));
    ASSERT_TRUE(Pc2_dec.isApprox(filterpy_Pc2_dec,0.001));
    ASSERT_TRUE(S2_dec.isApprox(filterpy_S2_dec,0.001));
    //----------------------------------------------------------------------
    //[CREATE]
    Estimator::EKF<Eigen::MatrixXd,
                   stateModel,
                   measureModelPol,
                   noiseTransitionModel,
                   stateModelJacobian,
                   measureModelJacobianPol> ekf_pol(x0,P0,Q0,R);

    Eigen::MatrixXd x_pol = ekf_pol.getState();
    Eigen::MatrixXd P_pol = ekf_pol.getCovariance();
    Eigen::MatrixXd Q_pol = ekf_pol.getGQG(t);
    Eigen::MatrixXd R_pol = ekf_pol.getMeasurementNoise();

    //FROM filterpy_test_ekf_pol.py
    Eigen::MatrixXd filterpy_x_pol(4,1);
    filterpy_x_pol << 20., 2., 20., 2.;
    Eigen::MatrixXd filterpy_P_pol(4,4);
    filterpy_P_pol << 1., 0., 0., 0.,
                      0., 1., 0., 0.,
                      0., 0., 1., 0.,
                      0., 0., 0., 1.;
    Eigen::MatrixXd filterpy_Q_pol(4,4);
    filterpy_Q_pol << 0.0004 ,0.004,      0.,    0.,
                       0.004 , 0.04,      0.,    0.,
                           0.,   0., 0.0004 , 0.004,
                           0.,   0.,  0.004 ,  0.04;
    Eigen::MatrixXd filterpy_R_pol(2,2);
    filterpy_R_pol << 1.,0.,
                      0.,1.;

    ASSERT_TRUE(x_pol.isApprox(filterpy_x_pol,0.001));
    ASSERT_TRUE(P_pol.isApprox(filterpy_P_pol,0.001));
    ASSERT_TRUE(Q_pol.isApprox(filterpy_Q_pol,0.001));
    ASSERT_TRUE(R_pol.isApprox(filterpy_R_pol,0.001));

    //[STEP-1]
    auto zS1_pol = ekf_pol.getMeasurementPredictData(t);
    Eigen::MatrixXd S1_pol = zS1_pol.second;

    auto pred_pol = ekf_pol.predict(t);
    Eigen::MatrixXd xp1_pol = pred_pol.first;
    Eigen::MatrixXd Pp1_pol = pred_pol.second;

    Eigen::MatrixXd filterpy_xp1_pol(4,1);
    filterpy_xp1_pol << 20.4, 2., 20.4, 2.;
    Eigen::MatrixXd filterpy_Pp1_pol(4,4);
    filterpy_Pp1_pol << 1.0404, 0.204,     0.,    0.,
                         0.204,  1.04,     0.,    0.,
                            0.,    0., 1.0404, 0.204,
                            0.,    0.,  0.204,  1.04;

    auto corr_pol = ekf_pol.correct(z1_pol);
    Eigen::MatrixXd xc1_pol = corr_pol.first;
    Eigen::MatrixXd Pc1_pol = corr_pol.second;

    Eigen::MatrixXd filterpy_xc1_pol(4,1);
    filterpy_xc1_pol << 22.74542798, 2.45988784, 22.7454077, 2.45988386;
    Eigen::MatrixXd filterpy_Pc1_pol(4,4);
    filterpy_Pc1_pol <<  0.77450057,  0.15186286, -0.26460055, -0.05188246,
                         0.15186286,  1.02977703, -0.05188246, -0.01017303,
                        -0.26460055, -0.05188246,  0.77450057,  0.15186286,
                        -0.05188246, -0.01017303,  0.15186286,  1.02977703;
    Eigen::MatrixXd filterpy_S1_pol(2,2);
    filterpy_S1_pol << 2.04040000e+00, 1.12879282e-18,
                       1.62186435e-18, 1.00125000e+00;
    Eigen::MatrixXd filterpy_K1_pol(2,4);
    filterpy_K1_pol << 0.36055376, -0.02546816,
                       0.07069682, -0.00499376,
                       0.36055376,  0.02546816,
                       0.07069682,  0.00499376;
    Eigen::MatrixXd filterpy_z1_pol(2,1);
    filterpy_z1_pol << 35.355, 0.785;

    //std::cout << "xc1_pol:" << std::endl << xc1_pol << std::endl;

    ASSERT_TRUE(xp1_pol.isApprox(filterpy_xp1_pol,0.001));
    ASSERT_TRUE(Pp1_pol.isApprox(filterpy_Pp1_pol,0.001));
    ASSERT_TRUE(xc1_pol.isApprox(filterpy_xc1_pol,0.001));
    ASSERT_TRUE(Pc1_pol.isApprox(filterpy_Pc1_pol,0.001));
    ASSERT_TRUE(S1_pol.isApprox(filterpy_S1_pol,0.001));

    //[STEP-2]
    auto zS2_pol = ekf_pol.getMeasurementPredictData(t);
    Eigen::MatrixXd S2_pol = zS2_pol.second;

    auto pred_pol2 = ekf_pol.predict(t);
    Eigen::MatrixXd xp2_pol = pred_pol2.first;
    Eigen::MatrixXd Pp2_pol = pred_pol2.second;

    Eigen::MatrixXd filterpy_xp2_pol(4,1);
    filterpy_xp2_pol << 23.23740555, 2.45988784, 23.23738447, 2.45988386;
    Eigen::MatrixXd filterpy_Pp2_pol(4,4);
    filterpy_Pp2_pol <<  0.8768368,  0.36181826, -0.28576046, -0.05391707,
                        0.36181826,  1.06977703, -0.05391707, -0.01017303,
                       -0.28576046, -0.05391707,   0.8768368,  0.36181826,
                       -0.05391707, -0.01017303,  0.36181826,  1.06977703;

    auto corr_pol2 = ekf_pol.correct(z2_pol);
    Eigen::MatrixXd xc2_pol = corr_pol2.first;
    Eigen::MatrixXd Pc2_pol = corr_pol2.second;

    Eigen::MatrixXd filterpy_xc2_pol(4,1);
    filterpy_xc2_pol << 24.08119142, 2.89862375, 24.0713034, 2.89509144;
    Eigen::MatrixXd filterpy_Pc2_pol(4,4);
    filterpy_Pc2_pol <<   0.76642064,  0.30440282, -0.39492619, -0.11088537,
                          0.30440282,  1.03990492, -0.11088534, -0.03988524,
                         -0.39492619, -0.11088534,  0.76642103,  0.30440299,
                         -0.11088537, -0.03988524,  0.30440299,  1.03990499;
    Eigen::MatrixXd filterpy_S2_pol(2,2);
    filterpy_S2_pol << 1.59107634e+00, -7.88691078e-09,
                      -7.88691078e-09,  1.00107653e+00;
    Eigen::MatrixXd filterpy_K2_pol(2,4);
    filterpy_K2_pol << 0.26268662, -0.02498874,
                       0.13683766, -0.00893577,
                       0.26268615,  0.02498876,
                       0.13683749,  0.00893578;
    Eigen::MatrixXd filterpy_z2_pol(2,1);
    filterpy_z2_pol << 36.056, 0.588;
    ASSERT_TRUE(xp2_pol.isApprox(filterpy_xp2_pol,0.001));
    ASSERT_TRUE(Pp2_pol.isApprox(filterpy_Pp2_pol,0.001));
    ASSERT_TRUE(xc2_pol.isApprox(filterpy_xc2_pol,0.001));
    ASSERT_TRUE(Pc2_pol.isApprox(filterpy_Pc2_pol,0.001));
    ASSERT_TRUE(S2_pol.isApprox(filterpy_S2_pol,0.001));
}

//TEST (EKF,ekf_ct_test) {
//    //----------------------------------------------------------------------
//    struct stateModel
//    {
//        Eigen::MatrixXd operator()(const Eigen::MatrixXd& x,double T)
//        {
//            double w = x(4);
//            double s = std::sin(w*T);
//            double ss = s/w;
//            double c = std::cos(w*T);
//            double cc = (1.-c)/w;

//            Eigen::MatrixXd F(5,5);
//            F << 1., ss, 0., -cc, 0.,
//                 0., c , 0., -s , 0.,
//                 0., cc, 1., ss , 0.,
//                 0., s , 0., c  , 0.,
//                 0., 0., 0., 0. , 1.;
//            return F*x;
//        }
//    };
//    struct stateModel_Jacobian
//    {
//        Eigen::MatrixXd operator()(const Eigen::MatrixXd& x,double T)
//        {
//            double w = x(4);
//            double vx = x(1);
//            long double vy = x(3);

//            double s = std::sin(w*T);
//            long double ss = s/w;
//            long double c = std::cos(w*T);
//            long double cc = (1.-c)/w;

//            double w0 = (T*vx*std::cos(w*T)/w) - (T*vy*std::sin(w*T)/w) - (vx*std::sin(w*T)/std::pow(w,2)) - (vy*(std::cos(w*T)-1.)/std::pow(w,2));
//            double w1 = -T*vx*std::sin(w*T) - T*vy*std::cos(w*T);
//            double w2 = (T*vx*std::sin(w*T)/w) + (T*vy*std::cos(w*T)/w) - (vx*(1.-std::cos(w*T))/std::pow(w,2)) - (vy*std::sin(w*T)/std::pow(w,2));
//            double w3 = T*vx*std::cos(w*T) - T*vy*std::sin(w*T);

//            Eigen::MatrixXd F(5,5);
//            F << 1., ss, -cc, 0., w0,
//                 0., c , -s , 0., w1,
//                 0., cc, ss , T , w2,
//                 0., s , c  , 1., w3,
//                 0., 0., 0. , 0., 1.                                                                                                                       ;
//            return F;
//        }
//    };
//    struct measureModel
//    {
//        Eigen::MatrixXd operator()(const Eigen::MatrixXd& x)
//        {
//            Eigen::MatrixXd H(2,5);
//            H << 1., 0., 0., 0., 0.,
//                 0., 0., 1., 0., 0.;
//            return H*x;
//        }
//        Eigen::MatrixXd operator()()
//        {
//            Eigen::MatrixXd H(2,5);
//            H << 1., 0., 0., 0., 0.,
//                 0., 0., 1., 0., 0.;
//            return H;
//        }
//    };
//    struct measureModel_Jacobian
//    {
//        Eigen::MatrixXd operator()(const Eigen::MatrixXd& x)
//        {
//            Eigen::MatrixXd H(2,5);
//            H << 1., 0., 0., 0., 0.,
//                 0., 0., 1., 0., 0.;
//            return H;
//        }
//        Eigen::MatrixXd operator()()
//        {
//            Eigen::MatrixXd H(2,5);
//            H << 1., 0., 0., 0., 0.,
//                 0., 0., 1., 0., 0.;
//            return H;
//        }
//    };
//    struct noiseTransitionModel
//    {
//        Eigen::MatrixXd operator()(double T)
//        {
//            Eigen::MatrixXd G(5,3);
//            G <<   T*T/2.,       0., 0.,
//                       T ,       0., 0.,
//                       0.,   T*T/2., 0.,
//                       0.,       T , 0.,
//                       0.,       0., 1.;
//            return G;
//        }
//    };
//    Eigen::MatrixXd x0(5,1);
//    x0 << 1.,2.,3.,4.,0.0001;
//    Eigen::MatrixXd R(2,2);
//    R << 1.,0.,
//         0.,1.;
//    Eigen::MatrixXd P0(5,5);
//    P0 << 1.,0.,0.,0.,0.,
//          0.,1.,0.,0.,0.,
//          0.,0.,1.,0.,0.,
//          0.,0.,0.,1.,0.,
//          0.,0.,0.,0.,1.;

//    Eigen::MatrixXd Q0(3,3);
//    Q0 << 1.,0.,0.,
//          0.,1.,0.,
//          0.,0.,1.;
//    Eigen::MatrixXd z1(2,1);
//    z1 << 10.,20.;
//    Eigen::MatrixXd z2(2,1);
//    z2 << 0.,40.;
//    Eigen::MatrixXd z3(2,1);
//    z3 << -20.,50.;

//    double T = 0.2;
//    //----------------------------------------------------------------------
//    Estimator::EKF<Eigen::MatrixXd,
//                   stateModel,
//                   measureModel,
//                   noiseTransitionModel,
//                   stateModel_Jacobian,
//                   measureModel_Jacobian> ekf(x0,P0,Q0,R);

//    auto pred1 = ekf.predict(T);
//    Eigen::MatrixXd xp1 = pred1.first;
//    Eigen::MatrixXd Pp1 = pred1.second;

//    auto corr1 = ekf.correct(z1);
//    Eigen::MatrixXd xc1 = corr1.first;
//    Eigen::MatrixXd Pc1 = corr1.second;
//    Eigen::MatrixXd Sc1 = ekf.getCovarianceOfMeasurementPredict();

//    auto pred2 = ekf.predict(T);
//    Eigen::MatrixXd xp2 = pred2.first;
//    Eigen::MatrixXd Pp2 = pred2.second;

//    auto corr2 = ekf.correct(z2);
//    Eigen::MatrixXd xc2 = corr2.first;
//    Eigen::MatrixXd Pc2 = corr2.second;
//    Eigen::MatrixXd Sc2 = ekf.getCovarianceOfMeasurementPredict();

//    auto pred3 = ekf.predict(T);
//    Eigen::MatrixXd xp3 = pred3.first;
//    Eigen::MatrixXd Pp3 = pred3.second;

//    auto corr3 = ekf.correct(z3);
//    Eigen::MatrixXd xc3 = corr3.first;
//    Eigen::MatrixXd Pc3 = corr3.second;
//    Eigen::MatrixXd Sc3 = ekf.getCovarianceOfMeasurementPredict();
//    //FROM filterpy_test_ekf.py
//    // init
//    Eigen::MatrixXd filterpy_x0(5,1);
//    filterpy_x0 << 1., 2., 3., 4., 1.e-04;
//    Eigen::MatrixXd filterpy_P0(5,5);
//    filterpy_P0 << 1., 0., 0., 0., 0.,
//                   0., 1., 0., 0., 0.,
//                   0., 0., 1., 0., 0.,
//                   0., 0., 0., 1., 0.,
//                   0., 0., 0., 0., 1.;
//    Eigen::MatrixXd filterpy_H(2,5);
//    filterpy_H << 1., 0., 0., 0., 0.,
//                  0., 0., 1., 0., 0.;
//    Eigen::MatrixXd filterpy_F(5,5);
//    filterpy_F << 1., 2.e-01        , 0., -2.00000017e-06, 0.,
//                  0., 1.            , 0., -2.e-05        , 0.,
//                  0., 2.00000017e-06, 1., 2.e-01         , 0.,
//                  0., 2.e-05        , 0., 1.             , 0.,
//                  0., 0.            , 0., 0.             , 1.;
//    Eigen::MatrixXd filterpy_R(2,2);
//    filterpy_R << 1., 0.,
//                  0., 1.;
//    Eigen::MatrixXd filterpy_Q(5,5);
//    filterpy_Q << 4.e-04, 4.e-03, 0.    , 0.    , 0.,
//                  4.e-03, 4.e-02, 0.    , 0.    , 0.,
//                  0.    , 0.    , 4.e-04, 4.e-03, 0.,
//                  0.    , 0.    , 4.e-03, 4.e-02, 0.,
//                  0.    , 0.    , 0.    , 0.    , 1.;

//    // step 1
//    Eigen::MatrixXd filterpy_xp1(5,1);
//    filterpy_xp1 << 1.399992e+00, 1.999920e+00, 3.800004e+00, 4.000040e+00, 1.000000e-04;
//    Eigen::MatrixXd filterpy_Pp1(5,5);
//    filterpy_Pp1 <<
//                    1.04680021,  0.26800234 ,-0.00319984, -0.03199597, -0.08000133,
//                      0.26800234,  1.6800256,  -0.0320005 , -0.3199808,  -0.800016  ,
//                     -0.00319984, -0.0320005 ,  0.08199979,  0.41999765,  0.03999733,
//                     -0.03199597, -0.3199808 ,  0.41999765,  2.1999744 ,  0.399968  ,
//                     -0.08000133, -0.800016  ,  0.03999733,  0.399968  ,  2.        ;
//    Eigen::MatrixXd filterpy_Sp1(2,2);
//    filterpy_Sp1 << 0., 0.,
//                   0., 0.;
//    Eigen::MatrixXd filterpy_xc1(5,1);
//    filterpy_xc1 <<  5.77488961 , 2.65273935 , 5.01523209 ,10.15842083 , 0.26143627;
//    Eigen::MatrixXd filterpy_Pc1(5,5);
//    filterpy_Pc1 <<
//                    5.11430272e-01,  1.30891595e-01, -1.44486599e-03, -1.50254223e-02,
//                      -3.90284356e-02,
//                      1.30891595e-01,  1.64401231e+00 ,-2.91882422e-02, -3.03533802e-01,
//                      -7.88377046e-01,
//                     -1.44486599e-03, -2.91882422e-02 , 7.57811269e-02,  3.88123527e-01,
//                       3.68506961e-02,
//                     -1.50254223e-02, -3.03533802e-01 , 3.88123527e-01,  2.03648268e+00,
//                       3.83242041e-01,
//                     -3.90284356e-02, -7.88377046e-01,  3.68506961e-02 , 3.83242041e-01,
//                       1.99540374e+00;
//    Eigen::MatrixXd filterpy_Sc1(2,2);
//    filterpy_Sc1 <<
//                     2.04680021, -0.00319984,
//                     -0.00319984,  1.08199979;

//    // step 2
//    Eigen::MatrixXd filterpy_xp2(5,1);
//    filterpy_xp2 <<
//      6.25209227,  2.1182,      7.05985791, 10.28317891,  0.26143627;
//    Eigen::MatrixXd filterpy_Pp2(5,5);
//    filterpy_Pp2 <<
//                    0.7969275,   1.89291231, -0.05266405, -0.33225292, -0.60987194,
//                      1.89291231, 13.55904706, -0.40200103, -2.61333092, -4.93161572,
//                     -0.05266405, -0.40200103,  0.12336005,  0.6374461,   0.15006819,
//                     -0.33225292, -2.61333092,  0.6374461,   3.3347794,   1.00853699,
//                    -0.60987194, -4.93161572,  0.15006819,  1.00853699,  2.99540374;
//    Eigen::MatrixXd filterpy_Sp2(2,2);
//    filterpy_Sp2 <<
//                     2.04680021 ,-0.00319984,
//                     -0.00319984 , 1.08199979;
//    Eigen::MatrixXd filterpy_xc2(5,1);
//    filterpy_xc2 <<
//          2.6235385,  -14.58630909,  10.80012246, 29.76820534,   6.24343147;
//    Eigen::MatrixXd filterpy_Pc2(5,5);
//    filterpy_Pc2 <<
//                     0.44272884 , 1.04436302, -0.02612533, -0.16850148, -0.33594346,
//                      1.04436302, 11.45798321, -0.3088953 , -2.06943416, -4.24833266,
//                     -0.02612533, -0.3088953 ,  0.10858868,  0.55954645,  0.11783937,
//                     -0.16850148, -2.06943416,  0.55954645,  2.92211359,  0.82180254,
//                     -0.33594346, -4.24833266,  0.11783937,  0.82180254,  2.77283731;
//    Eigen::MatrixXd filterpy_Sc2(2,2);
//    filterpy_Sc2 <<
//                    1.7969275,  -0.05266405,
//                     -0.05266405,  1.12336005;

//    // step 3
//    Eigen::MatrixXd filterpy_xp3(5,1);
//    filterpy_xp3 <<
//      -2.85111914, -32.85478346,  13.72615351,  -4.41244444,   6.24343147;
//    Eigen::MatrixXd filterpy_Pp3(5,5);
//    filterpy_Pp3 <<
//                      0.76070923,   3.18486479,  -0.12508075,  -0.12233364,   0.45602187,
//                      3.18486479, 100.09772967,  -7.65101444, -38.8159485  , 16.60109618,
//                     -0.12508075,  -7.65101444,   0.75231682,   3.96109818 , -1.28008185,
//                     -0.12233364, -38.8159485 ,   3.96109818,  21.69283138 , -6.6175288 ,
//                      0.45602187,  16.60109618,  -1.28008185,  -6.6175288  ,  3.77283731;
//    Eigen::MatrixXd filterpy_Sp3(2,2);
//    filterpy_Sp3 <<
//                     1.7969275,  -0.05266405,
//                    -0.05266405,  1.12336005;
//    Eigen::MatrixXd filterpy_xc3(5,1);
//    filterpy_xc3 <<
// -11.68866064, -213.16607222,   29.89275828,   76.25107043,  -23.28563619;
//    Eigen::MatrixXd filterpy_Pc3(5,5);
//    filterpy_Pc3 <<
//                     4.29152369e-01,  1.50631490e+00, -4.07472268e-02,  9.15698971e-02,
//                       2.08159220e-01,
//                     1.50631490e+00,  6.27168937e+01, -4.25870673e+00, -2.17625201e+01,
//                       1.04626905e+01,
//                    -4.07472268e-02, -4.25870673e+00,  4.26418396e-01,  2.26702829e+00,
//                      -7.15649772e-01,
//                     9.15698971e-02, -2.17625201e+01,  2.26702829e+00,  1.27241118e+01,
//                      -3.75730492e+00,
//                     2.08159220e-01,  1.04626905e+01, -7.15649772e-01, -3.75730492e+00,
//                       2.76182187e+00;
//    Eigen::MatrixXd filterpy_Sc3(2,2);
//    filterpy_Sc3 <<
//                     1.76070923, -0.12508075,
//                    -0.12508075,  1.75231682;

//    ASSERT_TRUE(x0.isApprox(filterpy_x0,0.00001));
//    ASSERT_TRUE(P0.isApprox(filterpy_P0,0.00001));

//    ASSERT_TRUE(xp1.isApprox(filterpy_xp1,0.00001));
//    ASSERT_TRUE(Pp1.isApprox(filterpy_Pp1,0.00001));
//    ASSERT_TRUE(xc1.isApprox(filterpy_xc1,0.00001));
//    ASSERT_TRUE(Pc1.isApprox(filterpy_Pc1,0.00001));
//    ASSERT_TRUE(Sc1.isApprox(filterpy_Sc1,0.00001));

//    ASSERT_TRUE(xp2.isApprox(filterpy_xp2,0.00001));
//    ASSERT_TRUE(Pp2.isApprox(filterpy_Pp2,0.00001));
//    ASSERT_TRUE(xc2.isApprox(filterpy_xc2,0.00001));
//    ASSERT_TRUE(Pc2.isApprox(filterpy_Pc2,0.00001));
//    ASSERT_TRUE(Sc2.isApprox(filterpy_Sc2,0.00001));

//    ASSERT_TRUE(xp3.isApprox(filterpy_xp3,0.00001));
//    ASSERT_TRUE(Pp3.isApprox(filterpy_Pp3,0.00001));
//    ASSERT_TRUE(xc3.isApprox(filterpy_xc3,0.00001));
//    ASSERT_TRUE(Pc3.isApprox(filterpy_Pc3,0.00001));
//    ASSERT_TRUE(Sc3.isApprox(filterpy_Sc3,0.00001));
//}
