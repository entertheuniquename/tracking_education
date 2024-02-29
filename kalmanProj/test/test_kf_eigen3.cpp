#include "test_kf_eigen3.h"

#include "data_creator_eigen3.h"
#include "printer_eigen3.h"

#include<unsupported/Eigen/MatrixFunctions>
#include<Eigen/Eigenvalues>

void progress_print(int am, int i, int per_step=1)
{
    static int c=0;if(i/(am/100.)>c){std::cout << "["+std::to_string(c)+"%]" << std::flush;c+=per_step;};
    if(i==am-1)std::cout << "["+std::to_string(c)+"%]" << std::endl;
}

int multix(const Eigen::MatrixXd a,
           const Eigen::MatrixXd b,
                 Eigen::MatrixXd& result)
{
    if(a.cols()==b.cols() && a.rows()==b.rows())
    {
        result.resize(a.rows(),a.cols());
        for(int k1=0;k1<a.rows();k1++)
            for(int k2=0;k2<a.cols();k2++)
                result(k1,k2) = a(k1,k2)*b(k1,k2);
        return 0;
    }
    else
        return 1;
};


Eigen::MatrixXd estimator_step(const Eigen::MatrixXd& measurements,
                               const Eigen::MatrixXd& P0,
                               const Eigen::MatrixXd& A,
                               const Eigen::MatrixXd& H,
                               const Eigen::MatrixXd& B,
                               const Eigen::MatrixXd& u,
                               const Eigen::MatrixXd& Q,
                               const Eigen::MatrixXd& G,
                               const Eigen::MatrixXd& R)
{
    KFE* kfe = new KFE(measurements.col(0),P0,A,Q,G,H,R);
    Eigen::MatrixXd estimations(measurements.rows(),measurements.cols()-1);
    estimations.setZero();
    for(int i=1;i<measurements.cols();i++)
    {
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> pred = kfe->predict(A,H,u,B);
        Eigen::MatrixXd zi(3,1);
        zi << measurements(0,i), measurements(2,i), measurements(4,i);
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> corr = kfe->correct(H,zi);
        estimations.col(i-1) = kfe->state;
    }
    return estimations;
};

Eigen::MatrixXd estimator_errors(const Eigen::MatrixXd& measurements,
                                 const Eigen::MatrixXd& estimations)
{
    Eigen::MatrixXd errors(estimations.rows(),estimations.cols());
    errors.setZero();
    Eigen::MatrixXd measurements0 = measurements.block(0,1,estimations.rows(),estimations.cols());
    multix(estimations-measurements0,estimations-measurements0,errors);
    return errors;
};

void test_KFE::data()
{

}

void test_KFE::estimation()
{
    // == make input data ==
    double meas_var = 4.;
    double max_speed = 4.;
    double process_var = 1.;
    double dt = 0.2;
    Eigen::MatrixXd A(6,6);
    A << 1., dt, 0., 0., 0., 0.,
          0., 1., 0., 0., 0., 0.,
          0., 0., 1., dt, 0., 0.,
          0., 0., 0., 1., 0., 0.,
          0., 0., 0., 0., 1., dt,
          0., 0., 0., 0., 0., 1.;
    Eigen::MatrixXd H(3,6);
    H << 1., 0., 0., 0., 0., 0.,
          0., 0., 1., 0., 0., 0.,
          0., 0., 0., 0., 1., 0.;
    Eigen::MatrixXd R(3,3);
    R << meas_var,      0.,       0.,
               0.,meas_var,       0.,
               0.,      0., meas_var;
    Eigen::MatrixXd Q(3,3);
    Q << process_var,         0.,          0.,
                   0.,process_var,          0.,
                   0.,         0., process_var;
    Eigen::MatrixXd G(6,3);
    G << dt*dt/2.,       0.,       0.,
                dt,       0.,       0.,
                0., dt*dt/2.,       0.,
                0.,       dt,       0.,
                0.,       0., dt*dt/2.,
                0.,       0.,       dt;
    Eigen::MatrixXd x0(1,6);
    x0 << 10., 2., 0., 0., 0., 0.;
    Eigen::MatrixXd P0(6,6);
    P0 << meas_var,                          0.,       0.,                         0.,       0.,                         0.,
                0., (max_speed/3)*(max_speed/3),       0.,                         0.,       0.,                         0.,
                0.,                          0., meas_var,                         0.,       0.,                         0.,
                0.,                          0.,       0.,(max_speed/3)*(max_speed/3),       0.,                         0.,
                0.,                          0.,       0.,                         0., meas_var,                         0.,
                0.,                          0.,       0.,                         0.,       0.,(max_speed/3)*(max_speed/3);
    Eigen::MatrixXd B(6,6);
    B.setZero();
    Eigen::MatrixXd u(6,1);
    u.setZero();

    std::vector<MTN2> mtn_vec;

    // == statistic ==
    int iterations_statistic = 2000;
    Eigen::MatrixXd var_err;

    for(int i=0;i<iterations_statistic;i++)
    {
        // == make measurements ==
        std::pair<Eigen::MatrixXd,Eigen::MatrixXd> out = make_data(x0,A,G*Q*G.transpose(),100,-1.,1.);
        if(i==0)
        {
            var_err.resize(out.second.rows(),out.second.cols()-1);
            var_err.setZero();
        }

        // == estimation ==
        Eigen::MatrixXd est = estimator_step(out.second,P0,A,H,B,u,Q,G,R);;
        Eigen::MatrixXd err = estimator_errors(out.second,est);

        if(i==0)
        {
            //---------------------------------------------------------------------------------------------------------
            Eigen::VectorXd x0 = out.first.row(0);
            Eigen::VectorXd y0 = out.first.row(2);
            Eigen::VectorXd z0 = out.first.row(4);
            Eigen::VectorXd x1 = out.second.row(0);
            Eigen::VectorXd y1 = out.second.row(2);
            Eigen::VectorXd z1 = out.second.row(4);
            Eigen::VectorXd xe = est.row(0);
            Eigen::VectorXd ye = est.row(2);
            Eigen::VectorXd ze = est.row(4);
            mtn_vec.push_back(MTN2(TYPE::LINE,std::to_string(i)+"-step - xy",std::vector<double>(y0.data(),y0.data()+y0.size()),std::vector<double>(x0.data(),x0.data()+x0.size())));
            mtn_vec.push_back(MTN2(TYPE::SCATTER,std::to_string(i)+"-step - xy",std::vector<double>(y1.data(),y1.data()+y1.size()),std::vector<double>(x1.data(),x1.data()+x1.size())));
            mtn_vec.push_back(MTN2(TYPE::LINE,std::to_string(i)+"-step - xy",std::vector<double>(ye.data(),ye.data()+ye.size()),std::vector<double>(xe.data(),xe.data()+xe.size())));

            mtn_vec.push_back(MTN2(TYPE::LINE,std::to_string(i)+"-step - xz",std::vector<double>(z0.data(),z0.data()+z0.size()),std::vector<double>(x0.data(),x0.data()+x0.size())));
            mtn_vec.push_back(MTN2(TYPE::SCATTER,std::to_string(i)+"-step - xz",std::vector<double>(z1.data(),z1.data()+z1.size()),std::vector<double>(x1.data(),x1.data()+x1.size())));
            mtn_vec.push_back(MTN2(TYPE::LINE,std::to_string(i)+"-step - xz",std::vector<double>(ze.data(),ze.data()+ze.size()),std::vector<double>(xe.data(),xe.data()+xe.size())));
            //---------------------------------------------------------------------------------------------------------
        }
        var_err+=err;

        progress_print(iterations_statistic,i,5);
    }

    var_err/=iterations_statistic;
    //---------------------------------------------------------------------------------------------------------
    Eigen::VectorXd err_x = var_err.row(0);
    Eigen::VectorXd err_vx = var_err.row(1);
    Eigen::VectorXd err_y = var_err.row(2);
    Eigen::VectorXd err_vy = var_err.row(3);
    Eigen::VectorXd err_z = var_err.row(4);
    Eigen::VectorXd err_vz = var_err.row(5);
    mtn_vec.push_back(MTN2(TYPE::LINE,"x",std::vector<double>(err_x.data(),err_x.data()+err_x.size())));
    mtn_vec.push_back(MTN2(TYPE::LINE,"vx",std::vector<double>(err_vx.data(),err_vx.data()+err_vx.size())));
    mtn_vec.push_back(MTN2(TYPE::LINE,"y",std::vector<double>(err_y.data(),err_y.data()+err_y.size())));
    mtn_vec.push_back(MTN2(TYPE::LINE,"vy",std::vector<double>(err_vy.data(),err_vy.data()+err_vy.size())));
    mtn_vec.push_back(MTN2(TYPE::LINE,"z",std::vector<double>(err_z.data(),err_z.data()+err_z.size())));
    mtn_vec.push_back(MTN2(TYPE::LINE,"vz",std::vector<double>(err_vz.data(),err_vz.data()+err_vz.size())));
    //---------------------------------------------------------------------------------------------------------
    print_charts_universal3(mtn_vec);
    //---------------------------------------------------------------------------------------------------------
}
