#include "bind_track.h"

#include<string>
#include<iostream>
#include<variant>
#include<tuple>

namespace py = pybind11;

Measurement<Eigen::MatrixXd> make_meas(const py::tuple& tu)
{
    std::tuple<double,Eigen::MatrixXd> a_tup = py::cast<std::tuple<double,Eigen::MatrixXd>>(tu);
    double t = std::get<0>(a_tup);
    Eigen::MatrixXd m = std::get<1>(a_tup);

    Measurement<Eigen::MatrixXd> meas{t,m};
    return meas;
}

class BindTrackEKFE_xyz_ct
{
private:
    //#CAUSE
    using M = Eigen::MatrixXd;
    using ModelState = Models::StateModel_CT_Deg<M>;
    using ModelMeas = Models::MeasureModel_XvXYvYZvZW_XYZ<M>;


    Track<M,
          Estimator::EKFE<M, ModelState, ModelMeas>,
          EstimatorInitEKFE_xyz_ct<M, ModelState, ModelMeas>,
          ModelState,
          ModelMeas> track;

    //Track<Eigen::MatrixXd,
    //Estimator::EKFE<Eigen::MatrixXd,Models::StateModel_CT<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZW_XYZ<Eigen::MatrixXd>>,
    //EstimatorInitEKFE_xyz_ct<Eigen::MatrixXd,Models::StateModel_CT<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZW_XYZ<Eigen::MatrixXd>>,
    //Models::StateModel_CT<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZW_XYZ<Eigen::MatrixXd>> track;
    //~
public:

    BindTrackEKFE_xyz_ct(const py::tuple& tu):
        track(make_meas(tu)){}

    py::tuple step1(const py::tuple& tu)
    {
        auto res = track.step(make_meas(tu));
        return py::make_tuple(res.first,res.second);
    }
    py::tuple step2(double t)
    {
        auto res = track.step(t);
        return py::make_tuple(res.first,res.second);
    }
    Eigen::MatrixXd GetState(){return track.GetState();}
    Eigen::MatrixXd GetCovariance(){return track.GetCovariance();}
};

class BindTrackEKF_xyz_ct
{
private:
    //#CAUSE
    using M = Eigen::MatrixXd;
    using ModelState = Models::StateModel_CT_Deg<M>;
    using ModelMeas = Models::MeasureModel_XvXYvYZvZW_XYZ<M>;
    using ModelStateJacobian = Models::StateModel_CT_Deg_Jacobian<M>;
    using ModelMeasJacobian = Models::MeasureModel_XvXYvYZvZW_XYZ_Jacobian<M>;
    using ModelG = Models::GModel_XvXYvYZvZW<M>;

    Track<M,
          Estimator::EKF<M, ModelState, ModelMeas,ModelG,ModelStateJacobian,ModelMeasJacobian>,
          EstimatorInitEKF_xyz_ct<M, ModelState,ModelMeas,ModelG,ModelStateJacobian,ModelMeasJacobian>,
          ModelState,
          ModelMeas> track;

    //Track<Eigen::MatrixXd,
    //Estimator::EKFE<Eigen::MatrixXd,Models::StateModel_CT<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZW_XYZ<Eigen::MatrixXd>>,
    //EstimatorInitEKFE_xyz_ct<Eigen::MatrixXd,Models::StateModel_CT<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZW_XYZ<Eigen::MatrixXd>>,
    //Models::StateModel_CT<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZW_XYZ<Eigen::MatrixXd>> track;
    //~
public:

    BindTrackEKF_xyz_ct(const py::tuple& tu):
        track(make_meas(tu)){}

    py::tuple step1(const py::tuple& tu)
    {
        auto res = track.step(make_meas(tu));
        return py::make_tuple(res.first,res.second);
    }
    py::tuple step2(double t)
    {
        auto res = track.step(t);
        return py::make_tuple(res.first,res.second);
    }
    Eigen::MatrixXd GetState(){return track.GetState();}
    Eigen::MatrixXd GetCovariance(){return track.GetCovariance();}
};

class BindTrackEKF_xyz_ct_rad
{
private:
    //#CAUSE
    using M = Eigen::MatrixXd;
    using ModelState = Models::StateModel_CT<M>;
    using ModelMeas = Models::MeasureModel_XvXYvYZvZW_XYZ<M>;
    using ModelStateJacobian = Models::StateModel_CT_Jacobian<M>;
    using ModelMeasJacobian = Models::MeasureModel_XvXYvYZvZW_XYZ_Jacobian<M>;
    using ModelG = Models::GModel_XvXYvYZvZW<M>;

    Track<M,
          Estimator::EKF<M, ModelState, ModelMeas,ModelG,ModelStateJacobian,ModelMeasJacobian>,
          EstimatorInitEKF_xyz_ct_rad<M, ModelState,ModelMeas,ModelG,ModelStateJacobian,ModelMeasJacobian>,
          ModelState,
          ModelMeas> track;

    //Track<Eigen::MatrixXd,
    //Estimator::EKFE<Eigen::MatrixXd,Models::StateModel_CT<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZW_XYZ<Eigen::MatrixXd>>,
    //EstimatorInitEKFE_xyz_ct<Eigen::MatrixXd,Models::StateModel_CT<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZW_XYZ<Eigen::MatrixXd>>,
    //Models::StateModel_CT<Eigen::MatrixXd>,Models::MeasureModel_XvXYvYZvZW_XYZ<Eigen::MatrixXd>> track;
    //~
public:

    BindTrackEKF_xyz_ct_rad(const py::tuple& tu):
        track(make_meas(tu)){}

    py::tuple step1(const py::tuple& tu)
    {
        auto res = track.step(make_meas(tu));
        return py::make_tuple(res.first,res.second);
    }
    py::tuple step2(double t)
    {
        auto res = track.step(t);
        return py::make_tuple(res.first,res.second);
    }
    Eigen::MatrixXd GetState(){return track.GetState();}
    Eigen::MatrixXd GetCovariance(){return track.GetCovariance();}
};

void bind_track(pybind11::module &m)
{
    py::class_<BindTrackEKFE_xyz_ct>(m, "BindTrackEKFE_xyz_ct")
        .def(py::init<const py::tuple&>())
        .def("step",&BindTrackEKFE_xyz_ct::step1)
        .def("step",&BindTrackEKFE_xyz_ct::step2)
        .def("getState",&BindTrackEKFE_xyz_ct::GetState)
        .def("getCov",&BindTrackEKFE_xyz_ct::GetCovariance);
    py::class_<BindTrackEKF_xyz_ct>(m, "BindTrackEKF_xyz_ct")
        .def(py::init<const py::tuple&>())
        .def("step",&BindTrackEKF_xyz_ct::step1)
        .def("step",&BindTrackEKF_xyz_ct::step2)
        .def("getState",&BindTrackEKF_xyz_ct::GetState)
        .def("getCov",&BindTrackEKF_xyz_ct::GetCovariance);
    py::class_<BindTrackEKF_xyz_ct_rad>(m, "BindTrackEKF_xyz_ct_rad")
        .def(py::init<const py::tuple&>())
        .def("step",&BindTrackEKF_xyz_ct_rad::step1)
        .def("step",&BindTrackEKF_xyz_ct_rad::step2)
        .def("getState",&BindTrackEKF_xyz_ct_rad::GetState)
        .def("getCov",&BindTrackEKF_xyz_ct_rad::GetCovariance);
}
