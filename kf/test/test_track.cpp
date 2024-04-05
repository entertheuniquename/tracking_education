#include "test_track.h"
#include "data_creator_eigen3.h"
#include "../source/models.h"

test_Track::matrices test_Track::data(double meas_var,
                                      double velo_var,
                                      double process_var,
                                      Eigen::MatrixXd stateModel,
                                      Eigen::MatrixXd measurementModel,
                                      Eigen::MatrixXd GModel,
                                      Eigen::MatrixXd HposModel,
                                      Eigen::MatrixXd HvelModel,
                                      Eigen::MatrixXd x0)
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
    m.x0 << x0;
    m.P0 = HposModel.transpose()*m.Rpos*HposModel + HvelModel.transpose()*m.Rvel*HvelModel;
    m.B.resize(6,6);
    m.B.setZero();
    m.u.resize(6,1);
    m.u.setZero();

    return m;
}

void test_Track::doit()
{
    Eigen::MatrixXd out_raw;
    Eigen::MatrixXd out_times;
    Eigen::MatrixXd out_noised_process;
    Eigen::MatrixXd out_noised_meas;
    Eigen::MatrixXd est_;
    Eigen::MatrixXd est;
    Eigen::MatrixXd err;
    Eigen::MatrixXd var_err;

    Eigen::MatrixXd x0(1,6);
    x0 << 10., 200., 0., 0., 0., 0.;
    //[x,vx,y,vy,z,vz]
    matrices data0 = data(300.,//meas_var
                          30.,//velo_var
                          1.,//process_var
                          Models::stateModel_3A<Eigen::MatrixXd>(6.),
                          Models::measureModel_3A<Eigen::MatrixXd>(),
                          Models::GModel_3A<Eigen::MatrixXd>(6.),
                          Models::HposModel_3A<Eigen::MatrixXd>(),
                          Models::HvelModel_3A<Eigen::MatrixXd>(),
                          x0);
    // == make measurements ==
    make_data(out_raw,out_times,out_noised_process,out_noised_meas,
              data0.x0,data0.A/*,measurementModel=H*/,data0.G,data0.Q,data0.Rpos,data0.H,6.,100,-1.,1.);

    //std::cout << "test_Track|" << "out_raw(" << out_raw.rows() << "," << out_raw.cols() << "):" << std::endl
    //                                         << out_raw << std::endl;
    //std::cout << "test_Track|" << "out_times(" << out_times.rows() << "," << out_times.cols() << "):" << std::endl
    //                                         << out_times << std::endl;
    //std::cout << "test_Track|" << "out_noised_meas(" << out_noised_meas.rows() << "," << out_noised_meas.cols() << "):" << std::endl
    //                                         << out_noised_meas << std::endl;

    // == make measurement(for track) ==
    std::vector<Measurement<Eigen::MatrixXd>> meases;
    for(int i=0;i<out_noised_meas.cols();i++)
    {
        Measurement<Eigen::MatrixXd> me{out_times(0,i),out_noised_meas.col(i),data0.Rpos};
        meases.push_back(me);
    }

    // == make track ==
    Track<Eigen::MatrixXd,KFE> track(meases[0],data0.P0,data0.A,data0.Q,data0.G,data0.H);

    // == step(for track) ==
    for(int i=1;i<meases.size();i++)
    {
        track.step(meases[i]);
        Utils::progress_print(meases.size()-1,i,5,"track run: ");
    }

}
