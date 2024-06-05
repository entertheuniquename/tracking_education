#include "../source/ekf_eigen3.h"
#include <gtest/gtest.h>
#include <armadillo>
TEST (EKFE, ExtendedKalmanFilterMath) {
    // init
    auto stateModel = [](const arma::mat& s, double t) {
        arma::mat F = {{1, t, 0, 0},
                       {0, 1, 0, 0},
                       {0, 0, 1, t},
                       {0, 0, 0, 1}};
        arma::mat sn = F*s;
        return sn;
    };

    auto stateModelE3 = [](const Eigen::MatrixXd& s, double t) {
        Eigen::MatrixXd F(4,4);
        F << 1, t, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, t,
             0, 0, 0, 1;
        Eigen::MatrixXd sn = F*s;
        return sn;
    };

    auto measureModel = [](const arma::mat& s) {
        double angle = atan2(s(2), s(0));
        double range = sqrt(s(0)*s(0) + s(2)*s(2));
        arma::mat r = trans(arma::mat{angle, range});
        return r;
    };

    auto measureModelE3 = [](const Eigen::MatrixXd& s,const Eigen::MatrixXd& z=Eigen::MatrixXd{}) {
        double angle = atan2(s(2), s(0));
        double range = sqrt(s(0)*s(0) + s(2)*s(2));
        Eigen::MatrixXd r(2,1);
        r << angle, range;
        //= trans(arma::mat{angle, range});
        return r;
    };

    double t = 0.2;
    arma::mat S  = Utils_A::cholPSD(diagmat(arma::mat{100, 1e3, 100, 1e3}));
    arma::mat x  = trans(arma::mat{35., 0., 45., 0.});
    arma::mat Qs = Utils_A::cholPSD(diagmat(arma::mat{0, .01, 0, .01}));
    arma::mat z  = arma::colvec{0.926815, 50.2618};
    arma::mat Rs = Utils_A::cholPSD(diagmat(arma::mat{2e-6, 1.}));

    using namespace Estimator;
    ExtendedKalmanFilterMath f;

    // predict
    auto pred = f.predict(Utils::AE(Qs), Utils::AE(x), Utils::AE(S), stateModelE3, nullptr, t);
    {
        /*arma::colvec*/Eigen::MatrixXd expectedPredState(4,1);
        expectedPredState << 35.,
                              0.,
                             45.,
                              0.;
        /*arma::mat*/Eigen::MatrixXd expectedPredSqrtCov(4,4);
        expectedPredSqrtCov << -11.8322,       0.,       0.,       0.,
                               -16.9031, -26.7263,       0.,       0.,
                                     0.,       0., -11.8322,       0.,
                                     0.,       0., -16.9031, -26.7263;

        /*arma::mat*/Eigen::MatrixXd expectedPredJacobian(4,4);
        expectedPredJacobian << 1.,0.2, 0., 0.,
                                0., 1., 0., 0.,
                                0., 0., 1.,0.2,
                                0., 0., 0., 1.;

        ASSERT_TRUE(pred.x.isApprox(expectedPredState,0.0001));
        ASSERT_TRUE(pred.S.isApprox(expectedPredSqrtCov,0.0001));
        ASSERT_TRUE(pred.dFdx.isApprox(expectedPredJacobian,0.0001));
    }

    // correct
    auto corr = f.correct(Utils::AE(z), Utils::AE(Rs), pred.x, pred.S, measureModelE3, nullptr,Utils::AE(z));
    {
        /*arma::colvec*/Eigen::MatrixXd expectedCorrState(4,1);
        expectedCorrState << 30.1194,
                            -6.97231,
                             40.3092,
                            -6.70121;
        /*arma::mat*/Eigen::MatrixXd expectedCorrSqrtCov(4,4);
        expectedCorrSqrtCov << 0.615061,            0,            0,            0,
                               0.878658,      26.7263,            0,            0,
                               0.777205, -6.04034e-20,     0.130612,            0,
                                1.11029, -8.62905e-20,     0.186588,      26.7263;

        ASSERT_TRUE(corr.first.isApprox(expectedCorrState,0.0001));
        ASSERT_TRUE(corr.second.isApprox(expectedCorrSqrtCov,0.0001));
    }

}
