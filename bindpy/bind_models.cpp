#include "models.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <carma>

void bind_models(py::module &m) {
    m.def("stateModel", [](py::array_t<double> x, double dt){
        arma::Mat<double> xm = carma::arr_to_mat(x);
        arma::Mat<double> r  = Models::stateModel<arma::Mat<double>>(xm, dt);
        return carma::mat_to_arr(r);
    },R"pbdoc(
          stateModel
      )pbdoc",
      py::arg("x"),
      py::arg("dt")
    );
    m.def("stateModel3A", [](py::array_t<double> x, double dt){
        arma::Mat<double> xm = carma::arr_to_mat(x);
        arma::Mat<double> r  = Models::stateModel3A<arma::Mat<double>>(xm, dt);
        return carma::mat_to_arr(r);
    },R"pbdoc(
          stateModel3A
      )pbdoc",
      py::arg("x"),
      py::arg("dt")
    );
    m.def("stateModel3B", [](py::array_t<double> x, double dt){
        arma::Mat<double> xm = carma::arr_to_mat(x);
        arma::Mat<double> r  = Models::stateModel3B<arma::Mat<double>>(xm, dt);
        return carma::mat_to_arr(r);
    },R"pbdoc(
          stateModel3B
      )pbdoc",
      py::arg("x"),
      py::arg("dt")
    );
    m.def("measureModel", [](py::array_t<double> x){
//        std::cout << "A";
          arma::Mat<double> xm = carma::arr_to_mat(x);
//          std::cout << "B";
          arma::Mat<double> r  = Models::measureModel<arma::Mat<double>>(xm);
//          std::cout << "C";
          return carma::mat_to_arr(r);
    },R"pbdoc(
            measureModel
        )pbdoc",
        py::arg("x")
    );
    m.def("measureModel3A", [](py::array_t<double> x){
          arma::Mat<double> xm = carma::arr_to_mat(x);
          arma::Mat<double> r  = Models::measureModel3A<arma::Mat<double>>(xm);
          return carma::mat_to_arr(r);
    },R"pbdoc(
            measureModel3A
        )pbdoc",
        py::arg("x")
    );
    m.def("measureModel3B", [](py::array_t<double> x){
          arma::Mat<double> xm = carma::arr_to_mat(x);
          arma::Mat<double> r  = Models::measureModel3B<arma::Mat<double>>(xm);
          return carma::mat_to_arr(r);
    },R"pbdoc(
            measureModel3B
        )pbdoc",
        py::arg("x")
    );
}
