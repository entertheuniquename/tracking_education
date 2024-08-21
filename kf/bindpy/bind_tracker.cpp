#include "bind_tracker.h"

namespace py = pybind11;

using MatrixType = Eigen::MatrixXd;
using FModelType = Models10::FCV<MatrixType>;
using HModelType = Models10::H<MatrixType>;
using GModelType = Models10::G<MatrixType>;
using MeasurementType = Tracker::Measurement3<MatrixType>;
using EstimatorType = Estimator::KF<MatrixType,FModelType,HModelType,GModelType>;
using EstimatorInitializatorType = Tracker::EstimatorInitializator10<MatrixType,EstimatorType,MeasurementType>;
using TrackType = Tracker::Track<MatrixType,MeasurementType,EstimatorInitializatorType>;

using IMM_MatrixType = Eigen::MatrixXd;
using IMM_F1ModelType = Models10::FCV<IMM_MatrixType>;
using IMM_F2ModelType = Models10::FCT<IMM_MatrixType>;
using IMM_F3ModelType = Models10::FCA<IMM_MatrixType>;
using IMM_FJ1ModelType = Models10::FCV_Jacobian<MatrixType>;
using IMM_FJ2ModelType = Models10::FCT_Jacobian<MatrixType>;
using IMM_FJ3ModelType = Models10::FCA_Jacobian<MatrixType>;
using IMM_HModelType = Models10::H<IMM_MatrixType>;
using IMM_HJModelType = Models10::H_Jacobian<IMM_MatrixType>;
using IMM_GModelType = Models10::G<IMM_MatrixType>;
using IMM_Estimator1Type = Estimator::KF<IMM_MatrixType,IMM_F1ModelType,IMM_HModelType,IMM_GModelType>;
using IMM_Estimator2Type = Estimator::EKF<IMM_MatrixType,IMM_F2ModelType,IMM_HModelType,IMM_GModelType,IMM_FJ2ModelType,IMM_HJModelType>;
using IMM_Estimator3Type = Estimator::KF<IMM_MatrixType,IMM_F3ModelType,IMM_HModelType,IMM_GModelType>;
using IMM_EstimatorType = Estimator::IMM<IMM_MatrixType,IMM_Estimator1Type,IMM_Estimator2Type,IMM_Estimator3Type>;
using IMM_MeasurementType = Tracker::Measurement3<IMM_MatrixType>;
using IMM_EstimatorInitializatorType = Tracker::EstimatorInitializator10_IMM3<IMM_MatrixType,IMM_EstimatorType,IMM_MeasurementType>;
using IMM_TrackType = Tracker::Track<IMM_MatrixType,IMM_MeasurementType,IMM_EstimatorInitializatorType>;

class BindTracker_10_KF_CV_XX
{
private:
    double time;
    double counter;
    Tracker::TrackerGNN<MatrixType,
                                EstimatorInitializatorType,
                                EstimatorType,
                                TrackType,
                                MeasurementType> tracker;
public:
    BindTracker_10_KF_CV_XX():time(0.)/*,counter(0)*/{}

    MatrixType step(MatrixType zs, double dt)
    {
        std::vector<MeasurementType> measurements;
        for(int i=0;i<zs.rows();i++)
            measurements.push_back(MeasurementType{time,zs(i,0),zs(i,1),zs(i,2),0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.});

        time+=dt;

        MatrixType estimates(zs.rows(),10);//#TEMP #ZAGL
        estimates.setZero();

        tracker.step(measurements,dt);

        for(int i=0;i<tracker.tracks.size();i++)
            estimates.row(i) = Utils::transpose(tracker.tracks.at(i)->getState());

        return estimates;
    }
};

class BindTracker_10_IMM3_XX
{
private:
    double time;
    double counter;
    Tracker::TrackerGNN<IMM_MatrixType,
                               IMM_EstimatorInitializatorType,
                               IMM_EstimatorType,
                               IMM_TrackType,
                               IMM_MeasurementType> tracker;
public:
    BindTracker_10_IMM3_XX():time(0.){}

    MatrixType step(MatrixType zs, double dt)
    {
        std::vector<MeasurementType> measurements;
        for(int i=0;i<zs.rows();i++)
            measurements.push_back(MeasurementType{time,zs(i,0),zs(i,1),zs(i,2),0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.});

        time+=dt;

        MatrixType estimates(zs.rows(),10);//#TEMP #ZAGL
        estimates.setZero();

        tracker.step(measurements,dt);

        for(int i=0;i<tracker.tracks.size();i++)
            estimates.row(i) = Utils::transpose(tracker.tracks.at(i)->getState());

        return estimates;
    }
};

void bind_tracker(pybind11::module &m)
{
    py::class_<BindTracker_10_KF_CV_XX>(m, "BindTracker_10_KF_CV_XX")
        .def(py::init())
        .def("step",&BindTracker_10_KF_CV_XX::step);
    py::class_<BindTracker_10_IMM3_XX>(m, "BindTracker_10_IMM3_XX")
        .def(py::init())
        .def("step",&BindTracker_10_IMM3_XX::step);
}
