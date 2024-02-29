

#set(AUTOGEN_BUILD_DIR ${BIN_DIR})

set(CMAKE_AUTOMOC ON)

include_directories("/usr/include/x86_64_linux_gnu/qt5")
include_directories("/usr/include/x86_64_linux_gnu/qt5/QtCore")
include_directories("/usr/include/x86_64_linux_gnu/qt5/QtCharts")
include_directories("/usr/include/x86_64_linux_gnu/qt5/QtTest")

#message("autogen build dir:" ${AUTOGEN_BUILD_DIR})

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
#add_library(kalmanPrototype SHARED ${ALL_SRC})
add_executable("tests_run" ${TEST_SRC})

#add_executable(test_kalmanProj main.cpp test_kf_eigen3.cpp)
#add_test(tests_run kalmanProj)

qt5_use_modules(${PROJECT_NAME} Charts Core Test)

#set(sources_list_test
#    main.cpp
#    test_1.cpp)

#set(test_libraries CharmCore ${QT_QTTEST_LIBRARY} ${QT_LIBRARIES})

#add_executable(test_kalmanProj ${sources_list_test})
target_link_libraries(tests_run Qt5::Test Qt5::Charts kalmanProj)
##target_compile_options(test_estimator PRIVATE -O0 -g3)

#enable_testing()
#add_subdirectory(test)


