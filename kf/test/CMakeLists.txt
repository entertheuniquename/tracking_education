message("kf_test[0]:")

set(CMAKE_AUTOMOC ON)

message("kf_test[1]: установка automoc")

include_directories("/usr/include/x86_64_linux_gnu/qt5")
include_directories("/usr/include/x86_64_linux_gnu/qt5/QtCore")
include_directories("/usr/include/x86_64_linux_gnu/qt5/QtCharts")
include_directories("/usr/include/x86_64_linux_gnu/qt5/QtTest")

message("kf_test[2]: включение необходимых дирректорий")

find_package(Qt5Core REQUIRED)
if(NOT Qt5Core_FOUND)
    message("Qt5Core is NOT found:")
endif()

find_package(Qt5Test REQUIRED)
if(NOT Qt5Test_FOUND)
    message("Qt5Test is NOT found:")
endif()

find_package(Qt5Widgets REQUIRED)
if(NOT Qt5Widgets_FOUND)
    message("Qt5Widgets is NOT found:")
endif()

find_package(Qt5Charts REQUIRED)
if(NOT Qt5Charts_FOUND)
    message("Qt5Charts is NOT found:")
endif()

message("kf_test[3]: поиск необходимых пакетов")

file(GLOB TEST_SRC ${TEST_DIR}/* .cpp)
add_executable("tests_run" ${TEST_SRC})

message("kf_test[4]: создание исполняемого файла")

target_link_libraries(tests_run Qt5::Test Qt5::Charts ${PROJECT_NAME})

message("kf_test[5]: ***")
