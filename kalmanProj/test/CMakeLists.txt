set(CMAKE_AUTOMOC ON)

include_directories("/usr/include/x86_64_linux_gnu/qt5")
include_directories("/usr/include/x86_64_linux_gnu/qt5/QtCore")
include_directories("/usr/include/x86_64_linux_gnu/qt5/QtCharts")
include_directories("/usr/include/x86_64_linux_gnu/qt5/QtTest")


find_package(Qt5Core REQUIRED)
if(Qt5Core_FOUND)
    message(STATUS "------------------------------------")
    message(STATUS "Qt5Core is FOUND:")
    message(STATUS "include: ${QT5CORE_INCLUDE_DIRS}")
    message(STATUS "libraries: ${QT5CORE_LIBRARIES}")
    message(STATUS "version: ${QT5CORE_VERSION_STRING}")
    message(STATUS "------------------------------------")
else(Qt5Core_FOUND)
    message("Qt5Core is NOT found:")
endif(Qt5Core_FOUND)

find_package(Qt5Test REQUIRED)
if(Qt5Test_FOUND)
    message(STATUS "------------------------------------")
    message(STATUS "Qt5Test is FOUND:")
    message(STATUS "include: ${QT5TEST_INCLUDE_DIRS}")
    message(STATUS "libraries: ${QT5TEST_LIBRARIES}")
    message(STATUS "version: ${QT5TEST_VERSION_STRING}")
    message(STATUS "------------------------------------")
else(Qt5Test_FOUND)
    message("Qt5Test is NOT found:")
endif(Qt5Test_FOUND)

find_package(Qt5Widgets REQUIRED)
if(Qt5Widgets_FOUND)
    message(STATUS "Qt5Widgets is FOUND:")
else(Qt5Widgets_FOUND)
    message("Qt5Widgets is NOT found:")
endif(Qt5Widgets_FOUND)

find_package(Qt5Charts REQUIRED)
if(Qt5Charts_FOUND)
    message(STATUS "------------------------------------")
    message(STATUS "Qt5Charts is FOUND:")
    message(STATUS "include: ${QT5CHARTS_INCLUDE_DIRS}")
    message(STATUS "libraries: ${QT5CHARTS_LIBRARIES}")
    message(STATUS "version: ${QT5CHARTS_VERSION_STRING}")
    message(STATUS "------------------------------------")
else(Qt5Charts_FOUND)
    message("Qt5Charts is NOT found:")
endif(Qt5Charts_FOUND)

file(GLOB TEST_SRC ${TEST_DIR}/* .cpp)
add_executable("tests_run" ${TEST_SRC})

qt5_use_modules(${PROJECT_NAME} Charts Core Test)

target_link_libraries(tests_run Qt5::Test Qt5::Charts kalmanProj)


