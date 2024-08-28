#include "../source/models.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

namespace py = pybind11;

void bind_models(py::module &m) {
    m.def("stateModel_CV", [](double dt){
        Models::StateModel_CV<Eigen::MatrixXd> sm;
        return sm(dt);});
    m.def("stateModel_CVx", [](Eigen::MatrixXd x,double dt){
        Models::StateModel_CV<Eigen::MatrixXd> sm;
        return sm(x,dt);});
    m.def("stateModel_CTx", [](Eigen::MatrixXd x,double dt){
        Models::StateModel_CT_Deg<Eigen::MatrixXd> sm;
        return sm(x,dt);});
    m.def("stateModel_CTx_rad", [](Eigen::MatrixXd x,double dt){
        Models::StateModel_CT<Eigen::MatrixXd> sm;
        return sm(x,dt);});
    m.def("stateModel_CT6", [](Eigen::MatrixXd x){
        std::cout << "stateModel_CT6" << std::endl;
        Models::StateModel_CT<Eigen::MatrixXd> sm;
        return sm(x,6.);});
    m.def("measureModel_XX", [](){
          Models::MeasureModel_XvXYvYZvZ_XYZ<Eigen::MatrixXd> mm;
          return mm();});
    m.def("measureModel_XXx", [](Eigen::MatrixXd x){
          Models::MeasureModel_XvXYvYZvZ_XYZ<Eigen::MatrixXd> mm;
          return mm(x);});
    m.def("measureModel_XwX", [](){
          Models::MeasureModel_XvXYvYZvZW_XYZ<Eigen::MatrixXd> mm;
          return mm();});
    m.def("measureModel_XwXx", [](Eigen::MatrixXd x){
          Models::MeasureModel_XvXYvYZvZW_XYZ<Eigen::MatrixXd> mm;
          return mm(x);});
    m.def("measureModel_XRx", [](Eigen::MatrixXd x){
          Models::MeasureModel_XvXYvYZvZ_EAR<Eigen::MatrixXd> mm;
          return mm(x); });
    m.def("jacobian_ct", [](Eigen::MatrixXd x, double t){
          Models::Jacobian_CT<Eigen::MatrixXd> mm;
          return mm(x,t); });
}
