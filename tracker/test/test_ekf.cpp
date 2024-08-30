#include "../source/ekf.h"
#include "../source/models.h"
#include <gtest/gtest.h>

TEST (EKF,ekf_base_test) {
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
    struct stateModel_Jacobian
    {
        Eigen::MatrixXd operator()(const Eigen::MatrixXd& x,double T)
        {
            Eigen::MatrixXd F(4,4);
            F << 1., T , 0., 0.,
                 0., 1., 0., 0.,
                 0., 0., 1., T ,
                 0., 0., 0., 1.;
            return F;
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
    struct measureModel_Jacobian
    {
        Eigen::MatrixXd operator()(const Eigen::MatrixXd& x)
        {
            Eigen::MatrixXd H(2,4);
            H << 1., 0., 0., 0.,
                 0., 0., 1., 0.;
            return H;
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

    Estimator::EKFMath<Eigen::MatrixXd> ekf_math;
    Estimator::EKF<Eigen::MatrixXd,
                   stateModel,
                   measureModel,
                   noiseTransitionModel,
                   stateModel_Jacobian,
                   measureModel_Jacobian> ekf(x0,P0,Q0,R);

    auto zS = ekf.getMeasurementPredictData(t);

    auto pred = ekf.predict(t);
    Eigen::MatrixXd xp1 = pred.first;
    Eigen::MatrixXd Pp1 = pred.second;

    auto corr = ekf.correct(z);
    Eigen::MatrixXd xc1 = corr.first;
    Eigen::MatrixXd Pc1 = corr.second;
    Eigen::MatrixXd zp1 = ekf.getMeasurementPredict();
    Eigen::MatrixXd Sc1 = ekf.getCovarianceOfMeasurementPredict();

    //FROM filterpy_test_ekf.py
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

    ASSERT_TRUE(zp1.isApprox(zS.first,0.001));
    ASSERT_TRUE(Sc1.isApprox(zS.second,0.001));
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
