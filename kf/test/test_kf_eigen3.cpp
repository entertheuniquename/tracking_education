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
}


std::pair<Eigen::MatrixXd,Eigen::MatrixXd> estimator_step(const Eigen::MatrixXd& measurements,
                               const Eigen::MatrixXd& P0,
                               const Eigen::MatrixXd& A,
                               const Eigen::MatrixXd& H,
                               const Eigen::MatrixXd& B,
                               const Eigen::MatrixXd& u,
                               const Eigen::MatrixXd& Q,
                               const Eigen::MatrixXd& G,
                               const Eigen::MatrixXd& R)
{
    std::unique_ptr<KFE> kfe = std::make_unique<KFE>(H.transpose()*measurements.col(0),P0,A,Q,G,H,R);
    Eigen::MatrixXd estimations((H.transpose()*measurements).rows(),measurements.cols()-1);
    estimations.setZero();
    Eigen::MatrixXd estimations_((H.transpose()*measurements).rows(),measurements.cols()-1);
    estimations_.setZero();
    for(int i=1;i<measurements.cols();i++)
    {
        //std::cout << "________" << std::endl;
        Eigen::MatrixXd covariance_prev = kfe->get_covariance();
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> pred = kfe->predict(A,H,u,B);
        estimations_.col(i-1) = kfe->get_state();
        Eigen::MatrixXd zi = measurements.col(i);
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> corr = kfe->correct(H,zi);
        estimations.col(i-1) = kfe->get_state();
        if(eigen3_matrix_check(kfe->get_covariance()))
            kfe->get_covariance() = covariance_prev;
        //std::cout << "measurements:" << std::endl << measurements.col(i) << std::endl;
        //std::cout << "estimations_:" << std::endl << estimations_.col(i-1) << std::endl;// << "* " << pred.first << std::endl;
        //std::cout << "estimations:" << std::endl << estimations.col(i-1) << std::endl;// << "* " << corr.first << std::endl;
    }

    return std::make_pair(estimations_,estimations);
}

Eigen::MatrixXd estimator_errors(const Eigen::MatrixXd& measurements,
                                 const Eigen::MatrixXd& estimations)
{
    Eigen::MatrixXd errors(estimations.rows(),estimations.cols());
    errors.setZero();
    Eigen::MatrixXd measurements0 = measurements.block(0,1,estimations.rows(),estimations.cols());
    multix(estimations-measurements0,estimations-measurements0,errors);
    return errors;
}


Eigen::MatrixXd make_covariance0(double meas_var,double max_speed)
{
    Eigen::MatrixXd p(6,6);
    p << meas_var,                          0.,       0.,                          0.,       0.,                          0.,
               0., (max_speed/3)*(max_speed/3),       0.,                          0.,       0.,                          0.,
               0.,                          0., meas_var,                          0.,       0.,                          0.,
               0.,                          0.,       0., (max_speed/3)*(max_speed/3),       0.,                          0.,
               0.,                          0.,       0.,                          0., meas_var,                          0.,
               0.,                          0.,       0.,                          0.,       0., (max_speed/3)*(max_speed/3);

    return p;
}

Eigen::MatrixXd make_covariance(Eigen::MatrixXd Rpos,Eigen::MatrixXd Rvel)
{
    Eigen::MatrixXd Hpos(3,6);
    Hpos << 1., 0., 0., 0., 0., 0.,
            0., 0., 1., 0., 0., 0.,
            0., 0., 0., 0., 1., 0.;
    Eigen::MatrixXd Hvel(3,6);
    Hvel << 0., 1., 0., 0., 0., 0.,
            0., 0., 0., 1., 0., 0.,
            0., 0., 0., 0., 0., 1.;

    return Hpos.transpose()*Rpos*Hpos + Hvel.transpose()*Rvel*Hvel;
}

void detx(Eigen::MatrixXd X)
{
    if(X.determinant()==0)
        throw 66;
}

test_KFE::matrices test_KFE::data(
        double meas_var,
        double velo_var,
        double process_var,
        double dt,
        xvector x0)
{
    matrices m;
    m.A.resize(6,6);
    m.A << 1., dt, 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0.,
           0., 0., 1., dt, 0., 0.,
           0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 1., dt,
           0., 0., 0., 0., 0., 1.;
    m.H.resize(3,6);
    m.H << 1., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 1., 0.;
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
    m.G.resize(6,3);
    m.G << dt*dt/2.,       0.,       0.,
                 dt,       0.,       0.,
                 0., dt*dt/2.,       0.,
                 0.,       dt,       0.,
                 0.,       0., dt*dt/2.,
                 0.,       0.,       dt;
    m.x0.resize(1,6);
    m.x0 << x0.x, x0.vx, x0.y, x0.vy, x0.z, x0.vz;
    m.P0 = make_covariance(m.Rpos,m.Rvel);
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

void test_KFE::estimation()
{
    enum class MeasVec{X=0,Y=1,Z=2};
    enum class StateVec{X=0,VX=1,Y=2,VY=3,Z=4,VZ=5};
    //                    meas_var   velo_var process_var T    x0
    matrices data1 = data(40000000., 200000., 300.,        6., { 10.,200.,0.,0.,0.,0.});
    matrices data2 = data(       4.,      4.,   1.,       0.2, { 10.,  2.,0.,0.,0.,0.});
    matrices data3 = data(     300.,     30.,   1.,        6., {500.,200.,0.,0.,0.,0.});
    matrices data0 = data3;
    // ===============================================================================
    try
    {
        detx(data0.P0);
    }  catch (int x) {
        std::cout << "exception[" << std::to_string(x) << "]" << std::endl;
    }
    // ===============================================================================
    int amount = 100;

    std::vector<MTN2> mtn_vec;

    // == statistic ==
    int x_size = data0.x0.cols();
    int iterations_statistic = 2000;
    Eigen::MatrixXd var_err(x_size,amount-1);

    for(int i=0;i<iterations_statistic;i++)
    {
        // == make measurements ==
        std::pair<Eigen::MatrixXd,Eigen::MatrixXd> out = make_data(data0.x0,data0.A/*,measurementModel=H*/,data0.G,data0.Q,data0.Rpos,amount,-1.,1.);
        // == estimation ==
        std::pair<Eigen::MatrixXd,Eigen::MatrixXd> est = estimator_step(out.second,data0.P0,data0.A,data0.H,data0.B,data0.u,data0.Q,data0.G,data0.Rpos);
        Eigen::MatrixXd err = estimator_errors(data0.H.transpose()*out.second,est.second);

        if(i==0)
        {
            //-- charts -----------------------------------------------------------------------------------------------
            mtn_vec.push_back(MTN2(TYPE::LINE,std::to_string(i)+": y(x)",mrow(out.first,2),mrow(out.first,0),Qt::blue));
            mtn_vec.push_back(MTN2(TYPE::SCATTER,std::to_string(i)+": y(x)",mrow(out.second,1),mrow(out.second,0),Qt::green));
            mtn_vec.push_back(MTN2(TYPE::SCATTER,std::to_string(i)+": y(x)",mrow(est.first,2),mrow(est.first,0),Qt::gray));
            mtn_vec.push_back(MTN2(TYPE::LINE,std::to_string(i)+": y(x)",mrow(est.second,2),mrow(est.second,0),Qt::red));

            mtn_vec.push_back(MTN2(TYPE::LINE,std::to_string(i)+": z(x)",mrow(out.first,4),mrow(out.first,0),Qt::blue));
            mtn_vec.push_back(MTN2(TYPE::SCATTER,std::to_string(i)+": z(x)",mrow(out.second,2),mrow(out.second,0),Qt::green));
            mtn_vec.push_back(MTN2(TYPE::SCATTER,std::to_string(i)+": z(x)",mrow(est.first,4),mrow(est.first,0),Qt::gray));
            mtn_vec.push_back(MTN2(TYPE::LINE,std::to_string(i)+": z(x)",mrow(est.second,4),mrow(est.second,0),Qt::red));

            mtn_vec.push_back(MTN2(TYPE::LINE,std::to_string(i)+": err x",mrow(err,0)));
            mtn_vec.push_back(MTN2(TYPE::LINE,std::to_string(i)+": err y",mrow(err,2)));
            mtn_vec.push_back(MTN2(TYPE::LINE,std::to_string(i)+": err z",mrow(err,4)));
            //---------------------------------------------------------------------------------------------------------
        }
        var_err+=err;

        progress_print(iterations_statistic,i,5);
    }

    var_err/=iterations_statistic;
    //-- charts -----------------------------------------------------------------------------------------------
    mtn_vec.push_back(MTN2(TYPE::LINE,"err x",mrow(var_err,0)));
    mtn_vec.push_back(MTN2(TYPE::LINE,"err vx",mrow(var_err,1)));
    mtn_vec.push_back(MTN2(TYPE::LINE,"err y",mrow(var_err,2)));
    mtn_vec.push_back(MTN2(TYPE::LINE,"err vy",mrow(var_err,3)));
    mtn_vec.push_back(MTN2(TYPE::LINE,"err z",mrow(var_err,4)));
    mtn_vec.push_back(MTN2(TYPE::LINE,"err vz",mrow(var_err,5)));
    //---------------------------------------------------------------------------------------------------------
    print_charts_universal3(mtn_vec);
    //---------------------------------------------------------------------------------------------------------
}
