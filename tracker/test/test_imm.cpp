#include "../source/kf.h"
#include "../source/ekf.h"
#include "../source/imm.h"
#include "../source/measurement.h"
#include "../source/models.h"
#include <gtest/gtest.h>

TEST (IMM,imm_base_test_constructor0) {
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
    mu << 0.33333333, 0.33333333, 0.33333333;
    Eigen::MatrixXd trans(3,3);
    trans << 0.95, 0.025, 0.025,
             0.025, 0.95, 0.025,
             0.025, 0.025, 0.95;

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
                                 noiseTransitionModel>> imm1(mu,trans,kf01,kf02,kf03);

    //input data
    Eigen::MatrixXd x0_imm1 = imm1.getState();
    Eigen::MatrixXd P0_imm1 = imm1.getCovariance();
    std::vector<Eigen::MatrixXd> states0_imm1 = imm1.estimators_states();
    std::vector<Eigen::MatrixXd> covariances0_imm1 = imm1.estimators_covariances();
    std::vector<Eigen::MatrixXd> process_noises0_imm1 = imm1.estimators_process_noises(t);
    std::vector<Eigen::MatrixXd> measurement_noises0_imm1 = imm1.estimators_measurement_noises();
    Eigen::MatrixXd mu_in_imm1 = imm1.getMU();
    Eigen::MatrixXd tp_in_imm1 = imm1.getTP();

    //predict data
    auto pred_imm1 = imm1.predict(t);
    Eigen::MatrixXd xp_imm1 = pred_imm1.first;
    Eigen::MatrixXd Pp_imm1 = pred_imm1.second;
    std::vector<Eigen::MatrixXd> states_p_imm1 = imm1.estimators_states();
    std::vector<Eigen::MatrixXd> covariances_p_imm1 = imm1.estimators_covariances();

    //correct data
    auto corr_imm1 = imm1.correct(z);

    Eigen::MatrixXd xc_imm1 = corr_imm1.first;
    Eigen::MatrixXd Pc_imm1 = corr_imm1.second;
    std::vector<Eigen::MatrixXd> states_c_imm1 = imm1.estimators_states();
    std::vector<Eigen::MatrixXd> covariances_c_imm1 = imm1.estimators_covariances();

    //result data
    Eigen::MatrixXd mu_imm1 = imm1.getMU();

    //FROM filterpy_test_imm.py
    // input data
    Eigen::MatrixXd filterpy_x0(4,1);
    filterpy_x0 << 1., 2., 3., 4.;
    Eigen::MatrixXd filterpy_P0(4,4);
    filterpy_P0 << 1., 0., 0., 0.,
                   0., 1., 0., 0.,
                   0., 0., 1., 0.,
                   0., 0., 0., 1.;

    Eigen::MatrixXd filterpy_x0_kf1(4,1);
    filterpy_x0_kf1 << 1., 2., 3., 4.;
    Eigen::MatrixXd filterpy_P0_kf1(4,4);
    filterpy_P0_kf1 << 1., 0., 0., 0.,
                       0., 1., 0., 0.,
                       0., 0., 1., 0.,
                       0., 0., 0., 1.;
    Eigen::MatrixXd filterpy_Q0_kf1(4,4);
    filterpy_Q0_kf1 << 0.0004, 0.004,     0.,    0.,
                        0.004,  0.04,     0.,    0.,
                           0.,    0., 0.0004, 0.004,
                           0.,    0.,  0.004,  0.04;
    Eigen::MatrixXd filterpy_R0_kf1(2,2);
    filterpy_R0_kf1 << 1., 0.,
                       0., 1.;

    Eigen::MatrixXd filterpy_x0_kf2(4,1);
    filterpy_x0_kf2 << 1., 2., 3., 4.;
    Eigen::MatrixXd filterpy_P0_kf2(4,4);
    filterpy_P0_kf2 << 1., 0., 0., 0.,
                       0., 1., 0., 0.,
                       0., 0., 1., 0.,
                       0., 0., 0., 1.;
    Eigen::MatrixXd filterpy_Q0_kf2(4,4);
    filterpy_Q0_kf2 << 0., 0., 0., 0.,
                       0., 0., 0., 0.,
                       0., 0., 0., 0.,
                       0., 0., 0., 0.;
    Eigen::MatrixXd filterpy_R0_kf2(2,2);
    filterpy_R0_kf2 << 1., 0.,
                       0., 1.;

    Eigen::MatrixXd filterpy_x0_kf3(4,1);
    filterpy_x0_kf3 << 1., 2., 3., 4.;
    Eigen::MatrixXd filterpy_P0_kf3(4,4);
    filterpy_P0_kf3 << 1., 0., 0., 0.,
                       0., 1., 0., 0.,
                       0., 0., 1., 0.,
                       0., 0., 0., 1.;
    Eigen::MatrixXd filterpy_Q0_kf3(4,4);
    filterpy_Q0_kf3 << 0.004, 0.04,    0.,   0.,
                        0.04,  0.4,    0.,   0.,
                          0.,   0., 0.004, 0.04,
                          0.,   0.,  0.04,  0.4;
    Eigen::MatrixXd filterpy_R0_kf3(2,2);
    filterpy_R0_kf3 << 1., 0.,
                       0., 1.;

    Eigen::MatrixXd filterpy_mu0(1,3);
    filterpy_mu0 << 0.33333333, 0.33333333, 0.33333333;
    Eigen::MatrixXd filterpy_tp0(3,3);
    filterpy_tp0 << 0.95, 0.025, 0.025,
                      0.025, 0.95, 0.025,
                      0.025, 0.025, 0.95;
    //predict data
    Eigen::MatrixXd filterpy_xp(4,1);
    filterpy_xp << 1.4, 2., 3.8, 4.;
    Eigen::MatrixXd filterpy_Pp(4,4);
    filterpy_Pp << 1.04146667e+00, 2.14666667e-01, 9.20337723e-33, 4.60168861e-32,
                    2.14666667e-01, 1.14666667e+00, 1.31476818e-32, 2.62953635e-31,
                    9.20337723e-33, 1.31476818e-32, 1.04146667e+00, 2.14666667e-01,
                    4.60168861e-32, 2.62953635e-31, 2.14666667e-01, 1.14666667e+00;

    Eigen::MatrixXd filterpy_xp_kf1(4,1);
    filterpy_xp_kf1 << 1.4, 2., 3.8, 4.;
    Eigen::MatrixXd filterpy_Pp_kf1(4,4);
    filterpy_Pp_kf1 << 1.04040000e+00, 2.04000000e-01, 1.38050658e-32, 6.90253292e-32,
                      2.04000000e-01, 1.04000000e+00, 1.97215226e-32, 9.86076132e-32,
                      1.38050658e-32, 1.97215226e-32, 1.04040000e+00, 2.04000000e-01,
                      6.90253292e-32, 9.86076132e-32, 2.04000000e-01, 1.04000000e+00;
    Eigen::MatrixXd filterpy_xp_kf2(4,1);
    filterpy_xp_kf2 << 1.4, 2., 3.8, 4.;
    Eigen::MatrixXd filterpy_Pp_kf2(4,4);
    filterpy_Pp_kf2 << 1.04000000e+00, 2.00000000e-01, 1.38050658e-32, 6.90253292e-32,
                      2.00000000e-01, 1.00000000e+00, 1.97215226e-32, 9.86076132e-32,
                      1.38050658e-32, 1.97215226e-32, 1.04000000e+00, 2.00000000e-01,
                      6.90253292e-32, 9.86076132e-32, 2.00000000e-01, 1.00000000e+00;
    Eigen::MatrixXd filterpy_xp_kf3(4,1);
    filterpy_xp_kf3 << 1.4, 2., 3.8, 4.;
    Eigen::MatrixXd filterpy_Pp_kf3(4,4);
    filterpy_Pp_kf3 << 1.044, 0.24,    0.,   0.,
                       0.24,  1.4,    0.,   0.,
                         0.,   0., 1.044, 0.24,
                         0.,   0.,  0.24,  1.4;

    //correct data
    Eigen::MatrixXd filterpy_xc(4,1);
    filterpy_xc << 5.78761117, 2.90974556, 12.06503499, 5.71370675;
    Eigen::MatrixXd filterpy_Pc(4,4);
    filterpy_Pc << 0.510201720, 0.106074753, 0.0000270794591, 0.000547005074,
                    0.106074753, 1.14271000, 0.000547005074, 0.0110495025,
                    0.0000270794591, 0.000547005074, 0.510238355, 0.106814772,
                    0.000547005074, 0.0110495025, 0.106814772, 1.15765840;

    Eigen::MatrixXd filterpy_xc_kf1(4,1);
    filterpy_xc_kf1 << 5.78514017,  2.85983141, 12.06038032,  5.61968242;
    Eigen::MatrixXd filterpy_Pc_kf1(4,4);
    filterpy_Pc_kf1 << 5.09900020e-01, 9.99803960e-02, 3.31594908e-33, 3.31528589e-32,
            9.99803960e-02, 1.01960400e+00, 8.98906424e-33, 8.98726643e-32,
            3.31594908e-33, 8.98906424e-33, 5.09900020e-01, 9.99803960e-02,
            3.31528589e-32, 8.98726643e-32, 9.99803960e-02, 1.01960400e+00;
    Eigen::MatrixXd filterpy_xc_kf2(4,1);
    filterpy_xc_kf2 <<  5.78431373,  2.84313725, 12.05882353,  5.58823529;
    Eigen::MatrixXd filterpy_Pc_kf2(4,4);
    filterpy_Pc_kf2 << 5.09803922e-01, 9.80392157e-02, 3.31724958e-33, 3.31724958e-32,
            9.80392157e-02, 9.80392157e-01, 9.00396314e-33, 9.00396314e-32,
            3.31724958e-33, 9.00396314e-33, 5.09803922e-01, 9.80392157e-02,
            3.31724958e-32, 9.00396314e-32, 9.80392157e-02, 9.80392157e-01;
    Eigen::MatrixXd filterpy_xc_kf3(4,1);
    filterpy_xc_kf3 << 5.7925636,   3.00978474, 12.07436399,  5.90215264;
    Eigen::MatrixXd filterpy_Pc_kf3(4,4);
    filterpy_Pc_kf3 << 0.51076321, 0.11741683, 0.,         0. ,
            0.11741683, 1.37181996, 0. ,        0. ,
            0. ,        0.  ,       0.51076321, 0.11741683,
            0. ,        0.  ,       0.11741683, 1.37181996;

    //result data
    Eigen::MatrixXd filterpy_mu(1,3);
    filterpy_mu << 0.3186357, 0.31358825, 0.36777604;

    Eigen::MatrixXd filterpy_likelihood(1,3);
    filterpy_likelihood << 1.23344221e-37, 1.21390347e-37, 1.42366499e-37;

    Eigen::MatrixXd filterpy_Se_kf1(2,2);
    filterpy_Se_kf1 << 2.0404, 1.38050658e-32,
                    1.38050658e-32, 2.0404;

    Eigen::MatrixXd filterpy_Se_kf2(2,2);
    filterpy_Se_kf2 << 2.04, 1.38050658e-32,
                    1.38050658e-32, 2.04;

    Eigen::MatrixXd filterpy_Se_kf3(2,2);
    filterpy_Se_kf3 << 2.044, 0.,
                    0., 2.044;

    Eigen::MatrixXd filterpy_cbar(1,3);
    filterpy_cbar << 0.31973803, 0.31506913, 0.36519284;

    Eigen::MatrixXd filterpy_omega(3,3);
    filterpy_omega << 0.9467248, 0.025283, 0.02181284,
                       0.02451916, 0.94553483, 0.02146731,
                       0.02875605, 0.02918217, 0.95671986;

    //compare
    //std::cout << "*********************************************" << std::endl;
    //std::cout << "process_noises0_imm1.at(0)" << std::endl << process_noises0_imm1.at(0) << std::endl;
    //std::cout << "filterpy_Q0_kf1" << std::endl << filterpy_Q0_kf1 << std::endl;
    //std::cout << "*********************************************" << std::endl;

    //input data
    ASSERT_TRUE(x0_imm1.isApprox(filterpy_x0,0.00001));
    ASSERT_TRUE(P0_imm1.isApprox(filterpy_P0,0.00001));
    ASSERT_TRUE(imm1.estimators_amount() == 3);
    ASSERT_TRUE(states0_imm1.at(0).isApprox(filterpy_x0_kf1,0.00001));
    ASSERT_TRUE(covariances0_imm1.at(0).isApprox(filterpy_P0_kf1,0.00001));
    ASSERT_TRUE(process_noises0_imm1.at(0).isApprox(filterpy_Q0_kf1,0.00001));
    ASSERT_TRUE(measurement_noises0_imm1.at(0).isApprox(filterpy_R0_kf1,0.00001));
    ASSERT_TRUE(states0_imm1.at(1).isApprox(filterpy_x0_kf2,0.00001));
    ASSERT_TRUE(covariances0_imm1.at(1).isApprox(filterpy_P0_kf2,0.00001));
    ASSERT_TRUE(process_noises0_imm1.at(1).isApprox(filterpy_Q0_kf2,0.00001));
    ASSERT_TRUE(measurement_noises0_imm1.at(1).isApprox(filterpy_R0_kf2,0.00001));
    ASSERT_TRUE(states0_imm1.at(2).isApprox(filterpy_x0_kf3,0.00001));
    ASSERT_TRUE(covariances0_imm1.at(2).isApprox(filterpy_P0_kf3,0.00001));
    ASSERT_TRUE(process_noises0_imm1.at(2).isApprox(filterpy_Q0_kf3,0.00001));
    ASSERT_TRUE(measurement_noises0_imm1.at(2).isApprox(filterpy_R0_kf3,0.00001));
    ASSERT_TRUE(mu_in_imm1.isApprox(filterpy_mu0,0.00001));
    ASSERT_TRUE(tp_in_imm1.isApprox(filterpy_tp0,0.00001));

    //predict data
    ASSERT_TRUE(states_p_imm1.at(0).isApprox(filterpy_xp_kf1,0.00001));
    ASSERT_TRUE(covariances_p_imm1.at(0).isApprox(filterpy_Pp_kf1,0.00001));
    ASSERT_TRUE(states_p_imm1.at(1).isApprox(filterpy_xp_kf2,0.00001));
    ASSERT_TRUE(covariances_p_imm1.at(1).isApprox(filterpy_Pp_kf2,0.00001));
    ASSERT_TRUE(states_p_imm1.at(2).isApprox(filterpy_xp_kf3,0.00001));
    ASSERT_TRUE(covariances_p_imm1.at(2).isApprox(filterpy_Pp_kf3,0.00001));
    ASSERT_TRUE(xp_imm1.isApprox(filterpy_xp,0.00001));
    ASSERT_TRUE(Pp_imm1.isApprox(filterpy_Pp,0.00001));

    //correct data
    ASSERT_TRUE(states_c_imm1.at(0).isApprox(filterpy_xc_kf1,0.00001));
    ASSERT_TRUE(covariances_c_imm1.at(0).isApprox(filterpy_Pc_kf1,0.00001));
    ASSERT_TRUE(states_c_imm1.at(1).isApprox(filterpy_xc_kf2,0.00001));
    ASSERT_TRUE(covariances_c_imm1.at(1).isApprox(filterpy_Pc_kf2,0.00001));
    ASSERT_TRUE(states_c_imm1.at(2).isApprox(filterpy_xc_kf3,0.00001));
    ASSERT_TRUE(covariances_c_imm1.at(2).isApprox(filterpy_Pc_kf3,0.00001));
    ASSERT_TRUE(xc_imm1.isApprox(filterpy_xc,0.001));
    ASSERT_TRUE(Pc_imm1.isApprox(filterpy_Pc,0.001));

    //result data
    ASSERT_TRUE(mu_imm1.isApprox(filterpy_mu,0.001));
}

TEST (IMM,imm_base_test_constructor1) {
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

    Eigen::MatrixXd mu(1,3);
    mu << 0.33333333, 0.33333333, 0.33333333;
    Eigen::MatrixXd trans(3,3);
    trans << 0.95, 0.025, 0.025,
             0.025, 0.95, 0.025,
             0.025, 0.025, 0.95;

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
                                 noiseTransitionModel>> imm1(mu,trans,x0,P0,Q0,R);

    //input data
    Eigen::MatrixXd x0_imm1 = imm1.getState();
    Eigen::MatrixXd P0_imm1 = imm1.getCovariance();
    std::vector<Eigen::MatrixXd> states0_imm1 = imm1.estimators_states();
    std::vector<Eigen::MatrixXd> covariances0_imm1 = imm1.estimators_covariances();
    std::vector<Eigen::MatrixXd> process_noises0_imm1 = imm1.estimators_process_noises(t);
    std::vector<Eigen::MatrixXd> measurement_noises0_imm1 = imm1.estimators_measurement_noises();
    Eigen::MatrixXd mu_in_imm1 = imm1.getMU();
    Eigen::MatrixXd tp_in_imm1 = imm1.getTP();

    //predict data
    auto pred_imm1 = imm1.predict(t);
    Eigen::MatrixXd xp_imm1 = pred_imm1.first;
    Eigen::MatrixXd Pp_imm1 = pred_imm1.second;
    std::vector<Eigen::MatrixXd> states_p_imm1 = imm1.estimators_states();
    std::vector<Eigen::MatrixXd> covariances_p_imm1 = imm1.estimators_covariances();

    //correct data
    auto corr_imm1 = imm1.correct(z);

    Eigen::MatrixXd xc_imm1 = corr_imm1.first;
    Eigen::MatrixXd Pc_imm1 = corr_imm1.second;
    std::vector<Eigen::MatrixXd> states_c_imm1 = imm1.estimators_states();
    std::vector<Eigen::MatrixXd> covariances_c_imm1 = imm1.estimators_covariances();

    //result data
    Eigen::MatrixXd mu_imm1 = imm1.getMU();

    //FROM filterpy_test_imm.py
    // input data
    Eigen::MatrixXd filterpy_x0(4,1);
    filterpy_x0 << 1., 2., 3., 4.;
    Eigen::MatrixXd filterpy_P0(4,4);
    filterpy_P0 << 1., 0., 0., 0.,
                   0., 1., 0., 0.,
                   0., 0., 1., 0.,
                   0., 0., 0., 1.;

    Eigen::MatrixXd filterpy_x0_kf1(4,1);
    filterpy_x0_kf1 << 1., 2., 3., 4.;
    Eigen::MatrixXd filterpy_P0_kf1(4,4);
    filterpy_P0_kf1 << 1., 0., 0., 0.,
                       0., 1., 0., 0.,
                       0., 0., 1., 0.,
                       0., 0., 0., 1.;
    Eigen::MatrixXd filterpy_Q0_kf1(4,4);
    filterpy_Q0_kf1 << 0.0004, 0.004,     0.,    0.,
                        0.004,  0.04,     0.,    0.,
                           0.,    0., 0.0004, 0.004,
                           0.,    0.,  0.004,  0.04;
    Eigen::MatrixXd filterpy_R0_kf1(2,2);
    filterpy_R0_kf1 << 1., 0.,
                       0., 1.;

    Eigen::MatrixXd filterpy_x0_kf2(4,1);
    filterpy_x0_kf2 << 1., 2., 3., 4.;
    Eigen::MatrixXd filterpy_P0_kf2(4,4);
    filterpy_P0_kf2 << 1., 0., 0., 0.,
                       0., 1., 0., 0.,
                       0., 0., 1., 0.,
                       0., 0., 0., 1.;
    Eigen::MatrixXd filterpy_Q0_kf2(4,4);
    filterpy_Q0_kf2 << 0.0004 ,0.004,  0. ,    0. ,
            0.004 , 0.04  , 0. ,    0.   ,
            0.   ,  0.  ,   0.0004, 0.004 ,
            0.  ,   0.,     0.004 , 0.04 ;
    Eigen::MatrixXd filterpy_R0_kf2(2,2);
    filterpy_R0_kf2 << 1., 0.,
                       0., 1.;

    Eigen::MatrixXd filterpy_x0_kf3(4,1);
    filterpy_x0_kf3 << 1., 2., 3., 4.;
    Eigen::MatrixXd filterpy_P0_kf3(4,4);
    filterpy_P0_kf3 << 1., 0., 0., 0.,
                       0., 1., 0., 0.,
                       0., 0., 1., 0.,
                       0., 0., 0., 1.;
    Eigen::MatrixXd filterpy_Q0_kf3(4,4);
    filterpy_Q0_kf3 << 0.0004, 0.004,  0.   ,  0.,
            0.004 , 0.04,   0.   ,  0. ,
            0. ,    0.  ,   0.0004 ,0.004,
            0. ,    0.  ,   0.004,  0.04  ;
    Eigen::MatrixXd filterpy_R0_kf3(2,2);
    filterpy_R0_kf3 << 1., 0.,
                       0., 1.;

    Eigen::MatrixXd filterpy_mu0(1,3);
    filterpy_mu0 << 0.33333333, 0.33333333, 0.33333333;
    Eigen::MatrixXd filterpy_tp0(3,3);
    filterpy_tp0 << 0.95, 0.025, 0.025,
                      0.025, 0.95, 0.025,
                      0.025, 0.025, 0.95;
    //predict data
    Eigen::MatrixXd filterpy_xp(4,1);
    filterpy_xp << 1.4, 2.,  3.8, 4. ;
    Eigen::MatrixXd filterpy_Pp(4,4);
    filterpy_Pp << 1.04040000e+00, 2.04000000e-01 ,9.20337723e-33, 4.60168861e-32,
            2.04000000e-01, 1.04000000e+00, 1.31476818e-32 ,2.62953635e-31,
            9.20337723e-33, 1.31476818e-32, 1.04040000e+00 ,2.04000000e-01,
            4.60168861e-32, 2.62953635e-31, 2.04000000e-01 ,1.04000000e+00;

    Eigen::MatrixXd filterpy_xp_kf1(4,1);
    filterpy_xp_kf1 << 1.4, 2. , 3.8, 4. ;
    Eigen::MatrixXd filterpy_Pp_kf1(4,4);
    filterpy_Pp_kf1 << 1.04040000e+00, 2.04000000e-01, 1.38050658e-32, 6.90253292e-32,
            2.04000000e-01, 1.04000000e+00, 1.97215226e-32, 9.86076132e-32,
            1.38050658e-32, 1.97215226e-32, 1.04040000e+00, 2.04000000e-01,
            6.90253292e-32, 9.86076132e-32, 2.04000000e-01, 1.04000000e+00;
    Eigen::MatrixXd filterpy_xp_kf2(4,1);
    filterpy_xp_kf2 << 1.4 ,2. , 3.8, 4.;
    Eigen::MatrixXd filterpy_Pp_kf2(4,4);
    filterpy_Pp_kf2 << 1.04040000e+00, 2.04000000e-01, 1.38050658e-32, 6.90253292e-32,
            2.04000000e-01, 1.04000000e+00, 1.97215226e-32, 9.86076132e-32,
            1.38050658e-32, 1.97215226e-32, 1.04040000e+00, 2.04000000e-01,
            6.90253292e-32, 9.86076132e-32, 2.04000000e-01, 1.04000000e+00;
    Eigen::MatrixXd filterpy_xp_kf3(4,1);
    filterpy_xp_kf3 << 1.4, 2.,  3.8, 4. ;
    Eigen::MatrixXd filterpy_Pp_kf3(4,4);
    filterpy_Pp_kf3 << 1.0404, 0.204 , 0.,     0.    ,
            0.204,  1.04 ,  0. ,    0.    ,
            0.   ,  0.   ,  1.0404, 0.204 ,
            0.   ,  0.   ,  0.204 , 1.04  ;

    //correct data
    Eigen::MatrixXd filterpy_xc(4,1);
    filterpy_xc << 5.78514017,  2.85983141, 12.06038032,  5.61968242;
    Eigen::MatrixXd filterpy_Pc(4,4);
    filterpy_Pc << 5.09900020e-01, 9.99803960e-02, 1.05402517e-30, 5.48009176e-31,
            9.99803960e-02, 1.01960400e+00, 5.99270950e-33, 5.99151095e-32,
            1.05402517e-30, 5.99270950e-33, 5.09900020e-01, 9.99803960e-02,
            5.48009176e-31, 5.99151095e-32, 9.99803960e-02, 1.01960400e+00;

    Eigen::MatrixXd filterpy_xc_kf1(4,1);
    filterpy_xc_kf1 << 5.78514017,  2.85983141, 12.06038032,  5.61968242;
    Eigen::MatrixXd filterpy_Pc_kf1(4,4);
    filterpy_Pc_kf1 << 5.09900020e-01, 9.99803960e-02, 3.31594908e-33, 3.31528589e-32,
            9.99803960e-02, 1.01960400e+00, 8.98906424e-33, 8.98726643e-32,
            3.31594908e-33, 8.98906424e-33, 5.09900020e-01, 9.99803960e-02,
            3.31528589e-32, 8.98726643e-32, 9.99803960e-02, 1.01960400e+00;
    Eigen::MatrixXd filterpy_xc_kf2(4,1);
    filterpy_xc_kf2 <<   5.78514017,  2.85983141, 12.06038032,  5.61968242;
    Eigen::MatrixXd filterpy_Pc_kf2(4,4);
    filterpy_Pc_kf2 << 5.09900020e-01, 9.99803960e-02, 3.31594908e-33, 3.31528589e-32,
            9.99803960e-02, 1.01960400e+00, 8.98906424e-33, 8.98726643e-32,
            3.31594908e-33, 8.98906424e-33, 5.09900020e-01, 9.99803960e-02,
            3.31528589e-32, 8.98726643e-32, 9.99803960e-02, 1.01960400e+00;
    Eigen::MatrixXd filterpy_xc_kf3(4,1);
    filterpy_xc_kf3 <<  5.78514017,  2.85983141, 12.06038032,  5.61968242;
    Eigen::MatrixXd filterpy_Pc_kf3(4,4);
    filterpy_Pc_kf3 << 0.50990002 ,0.0999804 , 0.,         0.   ,
                        0.0999804,  1.019604 ,  0. ,        0.  ,
                            0.   ,      0. ,        0.50990002 ,0.0999804 ,
                            0.   ,      0. ,        0.0999804,  1.019604  ;

    //result data
    Eigen::MatrixXd filterpy_mu(1,3);
    filterpy_mu << 0.33333333, 0.33333333, 0.33333333;

    Eigen::MatrixXd filterpy_likelihood(1,3);
    filterpy_likelihood << 1.23344221e-37, 1.23344221e-37, 1.23344221e-37;

    Eigen::MatrixXd filterpy_Se_kf1(2,2);
    filterpy_Se_kf1 << 2.04040000e+00, 1.38050658e-32,
                       1.38050658e-32, 2.04040000e+00;

    Eigen::MatrixXd filterpy_Se_kf2(2,2);
    filterpy_Se_kf2 << 2.04040000e+00, 1.38050658e-32,
                       1.38050658e-32, 2.04040000e+00;

    Eigen::MatrixXd filterpy_Se_kf3(2,2);
    filterpy_Se_kf3 << 2.0404,     0.,
                           0., 2.0404;

    Eigen::MatrixXd filterpy_cbar(1,3);
    filterpy_cbar << 0.33333333, 0.33333333, 0.33333333;

    Eigen::MatrixXd filterpy_omega(3,3);
    filterpy_omega <<  0.95, 0.025, 0.025,
                      0.025,  0.95, 0.025,
                      0.025, 0.025,  0.95 ;

    //compare
    //std::cout << "*********************************************" << std::endl;
    //std::cout << "process_noises0_imm1.at(0)" << std::endl << process_noises0_imm1.at(0) << std::endl;
    //std::cout << "filterpy_Q0_kf1" << std::endl << filterpy_Q0_kf1 << std::endl;
    //std::cout << "*********************************************" << std::endl;

    //input data
    ASSERT_TRUE(x0_imm1.isApprox(filterpy_x0,0.00001));
    ASSERT_TRUE(P0_imm1.isApprox(filterpy_P0,0.00001));
    ASSERT_TRUE(imm1.estimators_amount() == 3);
    ASSERT_TRUE(states0_imm1.at(0).isApprox(filterpy_x0_kf1,0.00001));
    ASSERT_TRUE(covariances0_imm1.at(0).isApprox(filterpy_P0_kf1,0.00001));
    ASSERT_TRUE(process_noises0_imm1.at(0).isApprox(filterpy_Q0_kf1,0.00001));
    ASSERT_TRUE(measurement_noises0_imm1.at(0).isApprox(filterpy_R0_kf1,0.00001));
    ASSERT_TRUE(states0_imm1.at(1).isApprox(filterpy_x0_kf2,0.00001));
    ASSERT_TRUE(covariances0_imm1.at(1).isApprox(filterpy_P0_kf2,0.00001));
    ASSERT_TRUE(process_noises0_imm1.at(1).isApprox(filterpy_Q0_kf2,0.00001));
    ASSERT_TRUE(measurement_noises0_imm1.at(1).isApprox(filterpy_R0_kf2,0.00001));
    ASSERT_TRUE(states0_imm1.at(2).isApprox(filterpy_x0_kf3,0.00001));
    ASSERT_TRUE(covariances0_imm1.at(2).isApprox(filterpy_P0_kf3,0.00001));
    ASSERT_TRUE(process_noises0_imm1.at(2).isApprox(filterpy_Q0_kf3,0.00001));
    ASSERT_TRUE(measurement_noises0_imm1.at(2).isApprox(filterpy_R0_kf3,0.00001));
    ASSERT_TRUE(mu_in_imm1.isApprox(filterpy_mu0,0.00001));
    ASSERT_TRUE(tp_in_imm1.isApprox(filterpy_tp0,0.00001));

    //predict data
    ASSERT_TRUE(states_p_imm1.at(0).isApprox(filterpy_xp_kf1,0.00001));
    ASSERT_TRUE(covariances_p_imm1.at(0).isApprox(filterpy_Pp_kf1,0.00001));
    ASSERT_TRUE(states_p_imm1.at(1).isApprox(filterpy_xp_kf2,0.00001));
    ASSERT_TRUE(covariances_p_imm1.at(1).isApprox(filterpy_Pp_kf2,0.00001));
    ASSERT_TRUE(states_p_imm1.at(2).isApprox(filterpy_xp_kf3,0.00001));
    ASSERT_TRUE(covariances_p_imm1.at(2).isApprox(filterpy_Pp_kf3,0.00001));
    ASSERT_TRUE(xp_imm1.isApprox(filterpy_xp,0.00001));
    ASSERT_TRUE(Pp_imm1.isApprox(filterpy_Pp,0.00001));

    //correct data
    ASSERT_TRUE(states_c_imm1.at(0).isApprox(filterpy_xc_kf1,0.00001));
    ASSERT_TRUE(covariances_c_imm1.at(0).isApprox(filterpy_Pc_kf1,0.00001));
    ASSERT_TRUE(states_c_imm1.at(1).isApprox(filterpy_xc_kf2,0.00001));
    ASSERT_TRUE(covariances_c_imm1.at(1).isApprox(filterpy_Pc_kf2,0.00001));
    ASSERT_TRUE(states_c_imm1.at(2).isApprox(filterpy_xc_kf3,0.00001));
    ASSERT_TRUE(covariances_c_imm1.at(2).isApprox(filterpy_Pc_kf3,0.00001));
    ASSERT_TRUE(xc_imm1.isApprox(filterpy_xc,0.001));
    ASSERT_TRUE(Pc_imm1.isApprox(filterpy_Pc,0.001));

    //result data
    ASSERT_TRUE(mu_imm1.isApprox(filterpy_mu,0.001));
}

//TEST (IMM,imm3_test) {
//    //----------------------------------------------------------------------
//    using IMM_MatrixType = Eigen::MatrixXd;
//    using IMM_F1ModelType = Models10::FCV<IMM_MatrixType>;
//    using IMM_F2ModelType = Models10::FCT<IMM_MatrixType>;
//    using IMM_F3ModelType = Models10::FCA<IMM_MatrixType>;
//    using IMM_FJ1ModelType = Models10::FCV_Jacobian<IMM_MatrixType>;
//    using IMM_FJ2ModelType = Models10::FCT_Jacobian<IMM_MatrixType>;
//    using IMM_FJ3ModelType = Models10::FCA_Jacobian<IMM_MatrixType>;
//    using IMM_HModelType = Models10::H<IMM_MatrixType>;
//    using IMM_HJModelType = Models10::H_Jacobian<IMM_MatrixType>;
//    using IMM_GModelType = Models10::G<IMM_MatrixType>;
//    using IMM_Estimator1Type = Estimator::KF<IMM_MatrixType,IMM_F1ModelType,IMM_HModelType,IMM_GModelType>;
//    using IMM_Estimator2Type = Estimator::EKF<IMM_MatrixType,IMM_F2ModelType,IMM_HModelType,IMM_GModelType,IMM_FJ2ModelType,IMM_HJModelType>;
//    using IMM_Estimator3Type = Estimator::KF<IMM_MatrixType,IMM_F3ModelType,IMM_HModelType,IMM_GModelType>;
//    using IMM_EstimatorType = Estimator::IMM<IMM_MatrixType,IMM_Estimator1Type,IMM_Estimator2Type,IMM_Estimator3Type>;
//    using IMM_MeasurementType = Tracker::Measurement3<IMM_MatrixType>;
//    //----------------------------------------------------------------------
//    double dt = 6.;
//    IMM_MatrixType x0(10,1);
//    x0 << -154.678,
//          0.,
//          0.,
//    102.634,
//          0.,
//          0.,
//      596.7,
//          0.,
//          0.,
//          0.;
//    IMM_MatrixType P0(10,10);
//    P0 << 90000.,        0.,        0.,        0.,        0.,        0.,        0.,        0.,        0.,        0.,
//              0.,      900.,        0.,        0.,        0.,        0.,        0.,        0.,        0.,        0.,
//              0.,        0.,        9.,        0.,        0.,        0.,        0.,        0.,        0.,        0.,
//              0.,        0.,        0.,    90000.,        0.,        0.,        0.,        0.,        0.,        0.,
//              0.,        0.,        0.,        0.,      900.,        0.,        0.,        0.,        0.,        0.,
//              0.,        0.,        0.,        0.,        0.,        9.,        0.,        0.,        0.,        0.,
//              0.,        0.,        0.,        0.,        0.,        0.,    90000.,        0.,        0.,        0.,
//              0.,        0.,        0.,        0.,        0.,        0.,        0.,      900.,        0.,        0.,
//              0.,        0.,        0.,        0.,        0.,        0.,        0.,        0.,        9.,        0.,
//              0.,        0.,        0.,        0.,        0.,        0.,        0.,        0.,        0., 0.153664;
//    IMM_MatrixType Q0(4,4);
//    Q0 << 1., 0., 0., 0.,
//          0., 1., 0., 0.,
//          0., 0., 1., 0.,
//          0., 0., 0., 0.0001;
//    IMM_MatrixType R(3,3);
//    R << 90000.,     0.,     0.,
//             0., 90000.,     0.,
//             0.,     0., 90000.;

//    IMM_MatrixType z(3,1);
//    z << 1462.2,
//         98359.,
//        280.346;
//    //----------------------------------------------------------------------

//    Eigen::MatrixXd mu(1,3);
//    mu << 0.33333333, 0.33333333, 0.33333333;
//    Eigen::MatrixXd trans(3,3);
//    trans << 0.95, 0.025, 0.025,
//             0.025, 0.95, 0.025,
//             0.025, 0.025, 0.95;

//    Estimator::IMM<IMM_MatrixType,
//                   IMM_Estimator1Type,
//                   IMM_Estimator2Type,
//                   IMM_Estimator3Type> imm(mu,trans,x0,P0,Q0,R);

//    //input data
//    Eigen::MatrixXd x0_imm = imm.getState();
//    Eigen::MatrixXd P0_imm = imm.getCovariance();
//    std::vector<Eigen::MatrixXd> states0_imm = imm.estimators_states();
//    std::vector<Eigen::MatrixXd> covariances0_imm = imm.estimators_covariances();
//    std::vector<Eigen::MatrixXd> process_noises0_imm = imm.estimators_process_noises(dt);
//    std::vector<Eigen::MatrixXd> measurement_noises0_imm = imm.estimators_measurement_noises();
//    Eigen::MatrixXd mu_in_imm = imm.getMU();
//    Eigen::MatrixXd tp_in_imm = imm.getTP();

//    //predict data
//    auto pred_imm = imm.predict(dt);
//    Eigen::MatrixXd xp_imm = pred_imm.first;
//    Eigen::MatrixXd Pp_imm = pred_imm.second;
//    std::vector<Eigen::MatrixXd> states_p_imm = imm.estimators_states();
//    std::vector<Eigen::MatrixXd> covariances_p_imm = imm.estimators_covariances();

//    //correct data
//    auto corr_imm = imm.correct(z);

//    Eigen::MatrixXd xc_imm = corr_imm.first;
//    Eigen::MatrixXd Pc_imm = corr_imm.second;
//    std::vector<Eigen::MatrixXd> states_c_imm = imm.estimators_states();
//    std::vector<Eigen::MatrixXd> covariances_c_imm = imm.estimators_covariances();

//    //result data
//    Eigen::MatrixXd mu_imm = imm.getMU();

//    //compare
////    std::cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * *" << std::endl;
////    std::cout << "x0_imm:" << std::endl << x0_imm << std::endl;
////    std::cout << "P0_imm:" << std::endl << P0_imm << std::endl;
////    std::cout << "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _" << std::endl;
////    std::cout << "states0_imm.at(0):" << std::endl << states0_imm.at(0) << std::endl;
////    std::cout << "states0_imm.at(1):" << std::endl << states0_imm.at(1) << std::endl;
////    std::cout << "states0_imm.at(2):" << std::endl << states0_imm.at(2) << std::endl;
////    std::cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * *" << std::endl;
////    std::cout << "xp_imm:" << std::endl << xp_imm << std::endl;
////    std::cout << "Pp_imm:" << std::endl << Pp_imm << std::endl;
////    std::cout << "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _" << std::endl;
////    std::cout << "states_p_imm.at(0):" << std::endl << states_p_imm.at(0) << std::endl;
////    std::cout << "states_p_imm.at(1):" << std::endl << states_p_imm.at(1) << std::endl;
////    std::cout << "states_p_imm.at(2):" << std::endl << states_p_imm.at(2) << std::endl;
////    std::cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * *" << std::endl;
////    std::cout << "xc_imm:" << std::endl << xc_imm << std::endl;
////    std::cout << "Pc_imm:" << std::endl << Pc_imm << std::endl;
////    std::cout << "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _" << std::endl;
////    std::cout << "states_c_imm.at(0):" << std::endl << states_c_imm.at(0) << std::endl;
////    std::cout << "states_c_imm.at(1):" << std::endl << states_c_imm.at(1) << std::endl;
////    std::cout << "states_c_imm.at(2):" << std::endl << states_c_imm.at(2) << std::endl;
////    std::cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * *" << std::endl;
//}
