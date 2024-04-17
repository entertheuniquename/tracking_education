#pragma once

#include <pybind11-global/pybind11/pybind11.h>
#include <pybind11-global/pybind11/numpy.h>
#include <pybind11-global/pybind11/eigen.h>
#include "../source/kf_eigen3.h"

void bind_kf(pybind11::module &m);
