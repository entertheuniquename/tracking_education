#include "test_track.h"
#include "data_creator_eigen3.h"
#include "../source/models.h"
#include "../source/ekf_eigen3.h"//#TEMP
#include "exceptions.h"
#include "chprinter.h"

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
    //Eigen::MatrixXd var_err;

    double period = 6.;

    Eigen::MatrixXd x0(1,6);
    x0 << 10., 200., 0., 0., 0., 0.;
    //[x,vx,y,vy,z,vz]
    matrices data0 = data(300.,//meas_var
                          30.,//velo_var
                          1.,//process_var
                          Models::stateModel_3A<Eigen::MatrixXd>(period),
                          Models::measureModel_3A<Eigen::MatrixXd>(),
                          Models::GModel_3A<Eigen::MatrixXd>(period),
                          Models::HposModel_3A<Eigen::MatrixXd>(),
                          Models::HvelModel_3A<Eigen::MatrixXd>(),
                          x0);

    int measurement_amount = 100;
    int iterations_amount = 2000;

    double target_detection_probab = 0.9;

    Eigen::MatrixXd var_err(6,99);//#TEMP
    var_err.setZero();//#TEMP

    for(int i=0;i<iterations_amount;i++)
    {
        // == make pass indexes ==
        try{Exceptions::bad_prob_value(target_detection_probab);}  catch (int x) {
            std::cout << "exception[" << std::to_string(x) << "]" << std::endl;
        }
        int pass_am = (measurement_amount-1)*(1-target_detection_probab);

        std::vector<int> pass_index;
        for(int i=0;i<pass_am;i++)
            pass_index.push_back(Utils::rnd(1,measurement_amount));

        // == make measurements ==
        make_data(out_raw,out_times,out_noised_process,out_noised_meas,
                  data0.x0,data0.A/*,measurementModel=H*/,data0.G,data0.Q,data0.Rpos,data0.H,period,measurement_amount,-1.,1.);

        // == make measurement(for track) ==
        std::vector<Measurement<Eigen::MatrixXd>> meases;
        for(int i=0;i<out_noised_meas.cols();i++)
        {
            Measurement<Eigen::MatrixXd> me{out_times(0,i),out_noised_meas.col(i),data0.Rpos,data0.Q};
            meases.push_back(me);
        }

        // == make track ==
        Track<Eigen::MatrixXd,KFE> track(meases[0],data0.P0,data0.A,data0.Q,data0.G,data0.H);

        std::vector<Measurement<Eigen::MatrixXd>> estimates;
        // == step(for track) ==
        for(int i=1;i<meases.size();i++)
        {
            if(!(std::find(pass_index.begin(),pass_index.end(),i)!=pass_index.end()))
            {
                Measurement<Eigen::MatrixXd> m = track.step(meases[i]);
                estimates.push_back(m);
            }
            else
            {//пропуск
                Measurement<Eigen::MatrixXd> m = track.step(track.get_timepoint()+period);
                estimates.push_back(m);
            }
        }

        // == estimates vector ==
        est.resize(estimates[0].point.rows(),static_cast<int>(estimates.size()));
        for(int i=0;i<estimates.size();i++)
            est.col(i) = estimates[i].point;

        // == errors ==
        Eigen::MatrixXd measurements0 = out_noised_process.block(0,1,est.rows(),est.cols());
        Eigen::MatrixXd err =  Utils::mpow_obo(est-measurements0);

        // == var_err ==
        var_err+=err;

        Utils::progress_print(iterations_amount,i,5,"track statistic run: ");
    }

    var_err/=iterations_amount;

    std::vector<MTN2> mtn_vec;
    //-- charts -----------------------------------------------------------------------------------------------
    mtn_vec.push_back(MTN2(TYPE::LINE,"track-err x",Utils::mrow(var_err,0)));
    mtn_vec.push_back(MTN2(TYPE::LINE,"track-err vx",Utils::mrow(var_err,1)));
    mtn_vec.push_back(MTN2(TYPE::LINE,"track-err y",Utils::mrow(var_err,2)));
    mtn_vec.push_back(MTN2(TYPE::LINE,"track-err vy",Utils::mrow(var_err,3)));
    mtn_vec.push_back(MTN2(TYPE::LINE,"track-err z",Utils::mrow(var_err,4)));
    mtn_vec.push_back(MTN2(TYPE::LINE,"track-err vz",Utils::mrow(var_err,5)));
    //---------------------------------------------------------------------------------------------------------
    print_charts_universal3(mtn_vec);
}
