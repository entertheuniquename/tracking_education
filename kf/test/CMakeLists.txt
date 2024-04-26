message("kf_test[0]:")

set(CMAKE_AUTOMOC ON)

message("kf_test[1]: установка automoc")

include_directories("/usr/include/eigen3")
include_directories("/usr/include/x86_64_linux_gnu/qt5")
include_directories("/usr/include/x86_64_linux_gnu/qt5/QtCore")
include_directories("/usr/include/x86_64_linux_gnu/qt5/QtCharts")
include_directories("/usr/include/x86_64_linux_gnu/qt5/QtTest")

message("kf_test[2]: включение необходимых дирректорий")

file(GLOB TEST_SRC ${TEST_DIR}/* .cpp)
add_executable("tests_run" ${TEST_SRC})

message("kf_test[4]: создание исполняемого файла")

target_link_libraries(tests_run ${PROJECT_NAME})

message("kf_test[5]: ***")
