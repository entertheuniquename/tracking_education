message("kf_test[0]:")


set(CMAKE_AUTOMOC ON)

include_directories(${GTest_INCLUDES})
message("kf_test[1]: установка automoc")

include_directories("/usr/include/eigen3")

message("kf_test[2]: включение необходимых дирректорий")

file(GLOB TEST_SRC ${TEST_DIR}/* .cpp)
add_executable("tests_run" ${TEST_SRC})

message("kf_test[4]: создание исполняемого файла")

target_include_directories(tests_run PRIVATE ${Python3_INCLUDE_DIRS})

target_link_libraries(tests_run ${GTest_LIBRARIES} ${Python3_LIBRARIES} pthread gtest gtest_main armadillo)
target_compile_options(tests_run PRIVATE -O0 -g3)
message("kf_test[5]: ***")

include(GoogleTest)
gtest_discover_tests(tests_run)
