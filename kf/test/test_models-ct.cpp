#include <gtest/gtest.h>
#include <armadillo>

#include <Eigen/Dense>
#include <Eigen/QR>

#include "../source/ekf_eigen3.h"
#include "../source/models.h"

//TEST (EKF, test1) {
//    using M = arma::mat;

//    Estimator::ExtendedKalmanFilterModif<M,
//                                         Models::ConstTurn<M>,
//                                         Models::JacobianStateConstTurn<M>,
//                                         Models::TurnmeasCart<M>,
//                                         Models::JacobianMeasCartConstTurn<M>> ekf;

//    double dt = 6.;

//    ekf.InitialState = arma::trans(arma::mat{29341.1436070604,-33.5011725223503,29800.5152717619,-10.1432912663446,0,10000,0});
//    ekf.InitialStateCovariance.SetCovariance({{5820.09697375021,295.937134258485,0,0,0,0,0},{295.937134258485,115.047650894499,0,0,0,0,0},{0,0,5820.09697375021,295.937134258485,0,0,0},{0,0,295.937134258485,115.047650894499,0,0,0},{0,0,0,0,100.0036,0,0},{0,0,0,0,0,5820.09697375021,295.937134258485},{0,0,0,0,0,295.937134258485,115.047650894499}});
//    ekf.ProcessNoise.SetCovariance({{1,0,0,0},{0,1,0,0},{0,0,0.0001,0},{0,0,0,1}});

//    ekf.Predict(dt);

//    arma::mat ExpectedPredictState = arma::trans(arma::mat{29140.1365719263,-33.5011725223503,29739.6555241638,-10.1432912663446,0,10000,0});
//    arma::mat ExpectedPredictStateCovariance = {{14852.5422220745,1432.71777463233,-3353.93223489431,-1117.97741163144,318.672365047853,0,0},{1432.71777463233,263.879229230115,-1117.97741163144,-372.659137210479,106.224121682618,0,0},{-3353.93223489431,-1117.97741163144,24914.3956857729,4786.66892919845,-1052.50826376209,0,0},{-1117.97741163144,-372.659137210479,4786.66892919845,1381.86294741882,-350.836087920698,0,0},{318.672365047853,106.224121682618,-1052.50826376209,-350.836087920698,100.0072,0,0},{0,0,0,0,0,13837.058017054,1094.22303962548},{0,0,0,0,0,1094.22303962548,151.047650894499}};

//    double eps = 1e-8;

//    ASSERT_TRUE(Utils::TestMatAbsRel("PredictState",
//                                     ekf.PredictState,
//                                     ExpectedPredictState,
//                                     1e-9, eps));
//    ASSERT_TRUE(Utils::TestMatAbsRel("PredictStateCovariance",
//                                     ekf.PredictStateCovariance.GetCovariance(),
//                                     ExpectedPredictStateCovariance,
//                                     1e-9, eps));

//    ekf.MeasurementNoise.SetCovariance({{10000,0,0},{0,10000,0},{0,0,10000}});

//    arma::mat z = arma::trans(arma::mat{28116.1736319683,28744.1228391011,10000});
//    ekf.Correct(z);

//    arma::mat ExpectedCorrectState = arma::trans(arma::mat{28572.5857972954,-62.1130477086166,29073.1018203697,-116.588788943196,20.0807152227892,10000,0});
//    arma::mat ExpectedCorrectStateCovariance = {{5923.41867566164,540.278736150054,-391.602868768481,-268.305255276978,88.6928556445046,0,0},{540.278736150054,156.476612894141,-268.305255276979,-183.828352012927,60.7675815802357,0,0},{-391.602868768481,-268.305255276979,7098.2339091115,1345.19904254444,-292.93397778148,0,0},{-268.305255276978,-183.828352012927,1345.19904254444,707.964779883157,-200.702630027021,0,0},{88.6928556445046,60.7675815802357,-292.93397778148,-200.702630027021,66.3492605577208,0,0},{0,0,0,0,0,5804.85142384367,459.042822668229},{0,0,0,0,0,459.042822668229,100.81812762067}};

//    ASSERT_TRUE(Utils::TestMatAbsRel("CorrectState",
//                                     ekf.CorrectState,
//                                     ExpectedCorrectState,
//                                     1e-9, eps));

//    ASSERT_TRUE(Utils::TestMatAbsRel("CorrectStateCovariance",
//                                     ekf.CorrectStateCovariance.GetCovariance(),
//                                     ExpectedCorrectStateCovariance,
//                                     1e-9, eps));

//}

#define PRINTM(x) std::cerr << #x << std::endl << x << __FILE__ << ":" << __LINE__ << std::endl << std::endl


inline bool TestMatAbsRel(std::string name,
                          const arma::mat& Out,
                          const arma::mat& Expected,
                          const double abs_tol, /* small */
                          const double rel_tol  /* big   */) {

    if (!approx_equal(Out, Expected, "both", abs_tol, rel_tol)) {
       std::cout << name << std::endl;
       PRINTM(Out);
       PRINTM(Expected);

       arma::mat errAbsTol(Out.n_rows, Out.n_cols, arma::fill::zeros),
                 errRelTol(Out.n_rows, Out.n_cols, arma::fill::zeros);

       for (int r=0; r<int(Out.n_rows); ++r) {
           for (int c=0; c<int(Out.n_cols); ++c) {
                double errAbs = std::abs(Out(r, c) - Expected(r, c));
                double errRel = errAbs / std::max(std::abs(Out(r, c)), std::abs(Expected(r, c)));
                if (errAbs >= abs_tol) {
                    errAbsTol(r, c) = errAbs;
                }
                if (errRel>= rel_tol) {
                    errRelTol(r, c) = errRel;
                }
           }
       }

       PRINTM(errAbsTol);
       PRINTM(errRelTol);

       return false;
    }

    return true;
}


void convert(arma::mat& in) {
    arma::mat H = {
                    {1,0,0,0,0,0,0},
                    {0,1,0,0,0,0,0},
                    {0,0,1,0,0,0,0},
                    {0,0,0,1,0,0,0},
                    {0,0,0,0,0,1,0},
                    {0,0,0,0,0,0,1},
                    {0,0,0,0,1,0,0}
                  };
    if (in.n_cols==1) {
        in  = H * in;
    } else {
        in = H * in * trans(H);
    }
}


TEST(ctmodel, test1) {

    double dt = 6.;

    arma::mat InitialState = arma::trans(arma::mat{29341.1436070604,-33.5011725223503,29800.5152717619,-10.1432912663446,0,10000,0});
    arma::mat InitialStateCovariance {{5820.09697375021,295.937134258485,0,0,0,0,0},{295.937134258485,115.047650894499,0,0,0,0,0},{0,0,5820.09697375021,295.937134258485,0,0,0},{0,0,295.937134258485,115.047650894499,0,0,0},{0,0,0,0,100.0036,0,0},{0,0,0,0,0,5820.09697375021,295.937134258485},{0,0,0,0,0,295.937134258485,115.047650894499}};
    convert(InitialState);
    convert(InitialStateCovariance);

    arma::mat ProcessNoise {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,0.0001}};
    //ProcessNoise(3, 3) = std::pow(Utils::deg2rad(std::sqrt(ProcessNoise(3, 3))), 2.0);


    arma::mat dfdw (7, 4);
    double dt2 = pow(dt, 2);

    dfdw(0, 0) = dt2;
    dfdw(1, 0) = dt;
    dfdw(2, 1) = dt2;
    dfdw(3, 1) = dt;
    dfdw(4, 2) = dt2;
    dfdw(5, 2) = dt;
    dfdw(6, 3) = dt;


    arma::mat MeasurementNoise {{10000,0,0},{0,10000,0},{0,0,10000}};


    Estimator::EKFE<Eigen::MatrixXd,
                    Models::StateModel_CT_Deg<Eigen::MatrixXd>,
                    Models::MeasureModel_XvXYvYZvZW_XYZ<Eigen::MatrixXd>> ekf(Utils::AE(InitialState),
                                                                              Utils::AE(InitialStateCovariance),
                                                                              Utils::AE(ProcessNoise),
                                                                              Utils::AE(MeasurementNoise));
    double eps = 1e-8;
    auto pred = ekf.predict(dt);

    arma::mat ExpectedPredictState = arma::trans(arma::mat{29140.1365719263,-33.5011725223503,29739.6555241638,-10.1432912663446,0,10000,0});
    arma::mat ExpectedPredictStateCovariance = {{14852.5422220745,1432.71777463233,-3353.93223489431,-1117.97741163144,318.672365047853,0,0},{1432.71777463233,263.879229230115,-1117.97741163144,-372.659137210479,106.224121682618,0,0},{-3353.93223489431,-1117.97741163144,24914.3956857729,4786.66892919845,-1052.50826376209,0,0},{-1117.97741163144,-372.659137210479,4786.66892919845,1381.86294741882,-350.836087920698,0,0},{318.672365047853,106.224121682618,-1052.50826376209,-350.836087920698,100.0072,0,0},{0,0,0,0,0,13837.058017054,1094.22303962548},{0,0,0,0,0,1094.22303962548,151.047650894499}};
    convert(ExpectedPredictState);
    convert(ExpectedPredictStateCovariance);

    ASSERT_TRUE(TestMatAbsRel("PredictState",
                               Utils::EA(pred.first),
                               ExpectedPredictState,
                               1e-9, eps));
    ASSERT_TRUE(TestMatAbsRel("PredictStateCovariance",
                              Utils::EA(pred.second),
                              ExpectedPredictStateCovariance,
                              1e-9, eps));


    arma::mat z = arma::trans(arma::mat{28116.1736319683,28744.1228391011,10000});
    auto corr = ekf.correct(Utils::AE(z));

    arma::mat ExpectedCorrectState = arma::trans(arma::mat{28572.5857972954,-62.1130477086166,29073.1018203697,-116.588788943196,20.0807152227892,10000,0});
    arma::mat ExpectedCorrectStateCovariance = {{5923.41867566164,540.278736150054,-391.602868768481,-268.305255276978,88.6928556445046,0,0},{540.278736150054,156.476612894141,-268.305255276979,-183.828352012927,60.7675815802357,0,0},{-391.602868768481,-268.305255276979,7098.2339091115,1345.19904254444,-292.93397778148,0,0},{-268.305255276978,-183.828352012927,1345.19904254444,707.964779883157,-200.702630027021,0,0},{88.6928556445046,60.7675815802357,-292.93397778148,-200.702630027021,66.3492605577208,0,0},{0,0,0,0,0,5804.85142384367,459.042822668229},{0,0,0,0,0,459.042822668229,100.81812762067}};
    convert(ExpectedCorrectState);
    convert(ExpectedCorrectStateCovariance);

    ASSERT_TRUE(TestMatAbsRel("CorrectState",
                               Utils::EA(corr.first),
                               ExpectedCorrectState,
                               1e-9, eps));

    ASSERT_TRUE(TestMatAbsRel("CorrectStateCovariance",
                               Utils::EA(corr.second),
                               ExpectedCorrectStateCovariance,
                               1e-6, eps));
}


