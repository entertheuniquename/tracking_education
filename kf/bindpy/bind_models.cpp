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
    m.def("measureModel_XX", [](){
          Models::MeasureModel_XvXYvYZvZ_XYZ<Eigen::MatrixXd> mm;
          return mm();});
    m.def("measureModel_XXx", [](Eigen::MatrixXd x){
          Models::MeasureModel_XvXYvYZvZ_XYZ<Eigen::MatrixXd> mm;
          return mm(x);});
    m.def("measureModel_XRx", [](Eigen::MatrixXd x){
          Models::MeasureModel_XvXYvYZvZ_EAR<Eigen::MatrixXd> mm;
          return mm(x); });
}
