#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "../source/track.h"
#include "../source/models.h"

void bind_track(pybind11::module &m);
