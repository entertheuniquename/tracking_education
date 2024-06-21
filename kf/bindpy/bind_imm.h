#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "../source/models.h"
#include "../source/imm.h"
#include "../source/ekf2.h"//#TEMP - пока что считаю временным, для проверки проще поместить сюда

void bind_imm(pybind11::module &m);
