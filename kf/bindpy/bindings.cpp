#include "bind_kf.h"
#include "bind_models.h"

PYBIND11_MODULE(estimator, m) {
    bind_kf(m);
    bind_models(m);
}
