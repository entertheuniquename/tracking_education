set(LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../scripts)

pybind11_add_module(estimator
    MODULE
        bind_ekf_eigen3.cpp
        bind_measurement.cpp
        bind_track.cpp
        bind_models.cpp
        bindings.cpp
)

#CAUSE
target_link_libraries(estimator PRIVATE Eigen3::Eigen kf_compare)
#target_link_libraries(estimator PRIVATE Eigen3::Eigen kf)
#~

set_target_properties(estimator PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY})
