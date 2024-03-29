#include "test_kf_eigen3.h"
#include "data_creator_eigen3.h"
#include "chprinter.h"
#include<unsupported/Eigen/MatrixFunctions>
#include<Eigen/Eigenvalues>
#include "../source/models.h"

void progress_print(int am, int i, int per_step=1,std::string s="")
{
    static int c=0;
    static bool bs=false;
    if(!bs){std::cout << s;bs=true;}
    if(i/(am/100.)>c){std::cout << "["+std::to_string(c)+"%]" << std::flush;c+=per_step;};
    if(i==am-1){std::cout << "["+std::to_string(c)+"%]" << std::endl;c=0;bs=false;}
}

void detx(Eigen::MatrixXd X)
{
    if(X.determinant()==0)
        throw 66;
}
void bad_prob_value(double x)
{
    if(x<0 || x>1)
        throw 55;
}

Eigen::MatrixXd mpow_obo(Eigen::MatrixXd x)
{
    return (x.array()*x.array()).matrix();
}

std::pair<Eigen::MatrixXd,Eigen::MatrixXd> estimator_probab_step(const Eigen::MatrixXd& measurements,
                                                                 const Eigen::MatrixXd& P0,
                                                                 const Eigen::MatrixXd& A,
                                                                 const Eigen::MatrixXd& H,
                                                                 const Eigen::MatrixXd& B,
                                                                 const Eigen::MatrixXd& u,
                                                                 const Eigen::MatrixXd& Q,
                                                                 const Eigen::MatrixXd& G,
                                                                 const Eigen::MatrixXd& R,
                                                                 double target_detection_probab=1)
{
    // == make pass indexes ==
    try{bad_prob_value(target_detection_probab);}  catch (int x) {
        std::cout << "exception[" << std::to_string(x) << "]" << std::endl;
    }
    int pass_am = (measurements.cols()-1)*(1-target_detection_probab);

    std::vector<int> pass_index;
    for(int i=0;i<pass_am;i++)
        pass_index.push_back(rnd(1,measurements.cols()));
    //=====================================================
    // == steps ==
    std::unique_ptr<KFE> kfe = std::make_unique<KFE>(H.transpose()*measurements.col(0),P0,A,Q,G,H,R);
    Eigen::MatrixXd estimations((H.transpose()*measurements).rows(),measurements.cols()-1);
    estimations.setZero();
    Eigen::MatrixXd estimations_((H.transpose()*measurements).rows(),measurements.cols()-1);
    estimations_.setZero();
    for(int i=1;i<measurements.cols();i++)
    {
        Eigen::MatrixXd covariance_prev = kfe->get_covariance();
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> pred = kfe->predict(A,H,u,B);
        estimations_.col(i-1) = kfe->get_state();
        Eigen::MatrixXd zi = measurements.col(i);
        if(!(std::find(pass_index.begin(),pass_index.end(),i)!=pass_index.end()))
            std::pair<Eigen::MatrixXd, Eigen::MatrixXd> corr = kfe->correct(H,zi);

        estimations.col(i-1) = kfe->get_state();
        if(kfe->get_covariance().determinant()==0)
            kfe->get_covariance() = covariance_prev;
    }

    return std::make_pair(estimations_,estimations);
}

Eigen::MatrixXd estimator_errors(const Eigen::MatrixXd& measurements,
                                 const Eigen::MatrixXd& estimations)
{
    Eigen::MatrixXd measurements0 = measurements.block(0,1,estimations.rows(),estimations.cols());
    return mpow_obo(estimations-measurements0);
}

test_KFE::matrices test_KFE::data(
        double meas_var,
        double velo_var,
        double process_var,
        Eigen::MatrixXd stateModel,
        Eigen::MatrixXd measurementModel,
        Eigen::MatrixXd GModel,
        Eigen::MatrixXd HposModel,
        Eigen::MatrixXd HvelModel,
        xvector x0)
{
    matrices m;
    m.A = stateModel;
    m.H = measurementModel;
    m.Rpos.resize(3,3);
    m.Rpos << meas_var*meas_var,                0.,                0.,
                             0., meas_var*meas_var,                0.,
                             0.,                0., meas_var*meas_var;
    m.Rvel.resize(3,3);
    m.Rvel << velo_var*velo_var,               0.,                0.,
                             0.,velo_var*velo_var,                0.,
                             0.,               0., velo_var*velo_var;
    m.Q.resize(3,3);
    m.Q << process_var,         0.,          0.,
                    0.,process_var,          0.,
                    0.,         0., process_var;
    m.G = GModel;
    m.x0.resize(1,6);
    m.x0 << x0.x, x0.vx, x0.y, x0.vy, x0.z, x0.vz;
    m.P0 = HposModel.transpose()*m.Rpos*HposModel + HvelModel.transpose()*m.Rvel*HvelModel;
    m.B.resize(6,6);
    m.B.setZero();
    m.u.resize(6,1);
    m.u.setZero();

    return m;
}

std::vector<double> mrow(Eigen::MatrixXd m,int n)
{
    Eigen::VectorXd v = m.row(n);
    return  std::vector<double>(v.data(),v.data()+v.size());
}
std::vector<double> mcol(Eigen::MatrixXd m,int n)
{
    Eigen::VectorXd v = m.col(n);
    return  std::vector<double>(v.data(),v.data()+v.size());
}

void step(test_KFE::matrices& data0,
          Eigen::MatrixXd& out_raw,
          Eigen::MatrixXd& out_noised_process,
          Eigen::MatrixXd& out_noised_meas,
          Eigen::MatrixXd& est_,
          Eigen::MatrixXd& est,
          Eigen::MatrixXd& err,
          int measurement_amount)
{
    // == make measurements ==
    make_data(out_raw,out_noised_process,out_noised_meas,
              data0.x0,data0.A/*,measurementModel=H*/,data0.G,data0.Q,data0.Rpos,data0.H,measurement_amount,-1.,1.);
    // == estimation ==
    std::pair<Eigen::MatrixXd,Eigen::MatrixXd> est0 = estimator_probab_step(out_noised_meas,data0.P0,data0.A,data0.H,data0.B,data0.u,data0.Q,data0.G,data0.Rpos,0.9);
    est_ = est0.first;
    est = est0.second;
    err = estimator_errors(out_noised_process,est);
}

void stat(test_KFE::matrices data0,
          Eigen::MatrixXd& out_raw,
          Eigen::MatrixXd& out_noised_process,
          Eigen::MatrixXd& out_noised_meas,
          Eigen::MatrixXd& est_,
          Eigen::MatrixXd& est,
          Eigen::MatrixXd& err,
          Eigen::MatrixXd& var_err,
          int measurement_amount, int iterations_amount)
{
    var_err.resize(data0.x0.cols(),measurement_amount-1);
    var_err.setZero();

    for(int i=0;i<iterations_amount;i++)
    {
        out_raw.setZero();
        out_noised_process.setZero();
        out_noised_meas.setZero();
        est_.setZero();
        est.setZero();
        err.setZero();

        step(data0,out_raw,out_noised_process,out_noised_meas,est_,est,err,measurement_amount);

        var_err+=err;
        progress_print(iterations_amount,i,5,"statistic run: ");
    }
    var_err/=iterations_amount;
    var_err = sqrt_one_by_one(var_err);
}

template<class EX, class EZ>
void print_step(Eigen::MatrixXd& out_raw,
                Eigen::MatrixXd& out_noised_process,
                Eigen::MatrixXd& out_noised_meas,
                Eigen::MatrixXd& est_,
                Eigen::MatrixXd& est,
                std::string s="")
{
    //-- charts -----------------------------------------------------------------------------------------------
    std::vector<MTN2> mtn_vec;
    mtn_vec.push_back(MTN2(TYPE::LINE,s+"y(x)",mrow(out_raw,static_cast<int>(EX::Y)),mrow(out_raw,static_cast<int>(EX::X)),Qt::blue));
    mtn_vec.push_back(MTN2(TYPE::SCATTER,s+"y(x)",mrow(out_noised_meas,static_cast<int>(EZ::Y)),mrow(out_noised_meas,static_cast<int>(EZ::X)),Qt::green));
    mtn_vec.push_back(MTN2(TYPE::LINE,s+"y(x)",mrow(out_noised_process,static_cast<int>(EX::Y)),mrow(out_noised_process,static_cast<int>(EX::X)),Qt::magenta));
    mtn_vec.push_back(MTN2(TYPE::LINE,s+"y(x)",mrow(est,static_cast<int>(EX::Y)),mrow(est,static_cast<int>(EX::X)),Qt::red));

    mtn_vec.push_back(MTN2(TYPE::LINE,s+"z(x)",mrow(out_raw,static_cast<int>(EX::Z)),mrow(out_raw,static_cast<int>(EX::X)),Qt::blue));
    mtn_vec.push_back(MTN2(TYPE::SCATTER,s+"z(x)",mrow(out_noised_meas,static_cast<int>(EZ::Z)),mrow(out_noised_meas,static_cast<int>(EZ::X)),Qt::green));
    mtn_vec.push_back(MTN2(TYPE::LINE,s+"z(x)",mrow(out_noised_process,static_cast<int>(EX::Z)),mrow(out_noised_process,static_cast<int>(EX::X)),Qt::magenta));
    mtn_vec.push_back(MTN2(TYPE::LINE,s+"z(x)",mrow(est,static_cast<int>(EX::Z)),mrow(est,static_cast<int>(EX::X)),Qt::red));
    //---------------------------------------------------------------------------------------------------------
    print_charts_universal3(mtn_vec);
}

template<class EX>
void print_stat(Eigen::MatrixXd& var_err,
                std::string s="")
{
    std::vector<MTN2> mtn_vec;
    //-- charts -----------------------------------------------------------------------------------------------
    mtn_vec.push_back(MTN2(TYPE::LINE,s+"err x",mrow(var_err,static_cast<int>(EX::X))));
    mtn_vec.push_back(MTN2(TYPE::LINE,s+"err vx",mrow(var_err,static_cast<int>(EX::VX))));
    mtn_vec.push_back(MTN2(TYPE::LINE,s+"err y",mrow(var_err,static_cast<int>(EX::Y))));
    mtn_vec.push_back(MTN2(TYPE::LINE,s+"err vy",mrow(var_err,static_cast<int>(EX::VY))));
    mtn_vec.push_back(MTN2(TYPE::LINE,s+"err z",mrow(var_err,static_cast<int>(EX::Z))));
    mtn_vec.push_back(MTN2(TYPE::LINE,s+"err vz",mrow(var_err,static_cast<int>(EX::VZ))));
    //---------------------------------------------------------------------------------------------------------
    print_charts_universal3(mtn_vec);
}

void test_KFE::estimation()
{
    Eigen::MatrixXd out_raw;
    Eigen::MatrixXd out_noised_process;
    Eigen::MatrixXd out_noised_meas;
    Eigen::MatrixXd est_;
    Eigen::MatrixXd est;
    Eigen::MatrixXd err;
    Eigen::MatrixXd var_err;

    //================================================================================
    //[vz,y,z,vy,vx,x]
    matrices data1 = data(300.,//meas_var
                          30.,//velo_var
                          1.,//process_var
                          Models::stateModel_3B<Eigen::MatrixXd>(6.),
                          Models::measureModel_3B<Eigen::MatrixXd>(),
                          Models::GModel_3B<Eigen::MatrixXd>(6.),
                          Models::HposModel_3B<Eigen::MatrixXd>(),
                          Models::HvelModel_3B<Eigen::MatrixXd>(),
                          {0.,0.,0.,0.,200.,10.});

    try{detx(data1.P0);}  catch (int x) {
        std::cout << "exception[" << std::to_string(x) << "]" << std::endl;
    }

    stat(data1,out_raw,out_noised_process,out_noised_meas,est_,est,err,var_err,100,2000);

    print_step<Models::X3B,Models::Z>(out_raw,out_noised_process,out_noised_meas,est_,est,"[vz,y,z,vy,vx,x](X3B): ");
    print_stat<Models::X3B>(var_err,"[vz,y,z,vy,vx,x](X3B): ");
    //================================================================================
    //[x,vx,y,vy,z,vz]
    matrices data2 = data(300.,//meas_var
                          30.,//velo_var
                          1.,//process_var
                          Models::stateModel_3A<Eigen::MatrixXd>(6.),
                          Models::measureModel_3A<Eigen::MatrixXd>(),
                          Models::GModel_3A<Eigen::MatrixXd>(6.),
                          Models::HposModel_3A<Eigen::MatrixXd>(),
                          Models::HvelModel_3A<Eigen::MatrixXd>(),
                          {10.,200.,0.,0.,0.,0.});

    try{detx(data2.P0);}  catch (int x) {
        std::cout << "exception[" << std::to_string(x) << "]" << std::endl;
    }

    stat(data2,out_raw,out_noised_process,out_noised_meas,est_,est,err,var_err,100,2000);

    print_step<Models::X3A,Models::Z>(out_raw,out_noised_process,out_noised_meas,est_,est,"[x,vx,y,vy,z,vz](X3A): ");
    print_stat<Models::X3A>(var_err,"[x,vx,y,vy,z,vz](X3A): ");
    //================================================================================

}
