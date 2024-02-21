cmake_minimum_required(VERSION 3.14)
project(kalmanProj VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_CXX_FLAGS "-std=c++17 -pthread")

find_package(Armadillo REQUIRED)

if(Armadillo_FOUND)
    message(STATUS "------------------------------------")
    message(STATUS "Armadillo is FOUND:")
    message(STATUS "include: ${ARMADILLO_INCLUDE_DIRS}")
    message(STATUS "libraries: ${ARMADILLO_LIBRARIES}")
    message(STATUS "version: ${ARMADILLO_VERSION_STRING}")
    message(STATUS "------------------------------------")
else(Armadillo_FOUND)
    message("Armadillo is NOT found:")
endif(Armadillo_FOUND)

find_package(Eigen3 REQUIRED)

if(Eigen3_FOUND)
    message(STATUS "------------------------------------")
    message(STATUS "Eigen3 is FOUND:")
    message(STATUS "include: ${EIGEN3_INCLUDE_DIRS}")
    message(STATUS "libraries: ${EIGEN3_LIBRARIES}")
    message(STATUS "version: ${EIGEN3_VERSION_STRING}")
    message(STATUS "------------------------------------")
else(Eigen3_FOUND)
    message("Eigen3 is NOT found:")
endif(Eigen3_FOUND)

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

include_directories("/usr/include/x86_64_linux_gnu/qt5")
include_directories("/usr/include/eigen3")

message("cmake:[ " ${PROJECT_NAME} " ]")

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Установка дирректории для исполняемого файла
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR})

# Установка дирректории для файлов библиотек
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${BIN_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BIN_DIR})

file(GLOB ALL_SRC ${SOURCE_DIR}/* .cpp)
#add_library(kalmanPrototype SHARED ${ALL_SRC})
add_executable(${PROJECT_NAME} ${ALL_SRC})

qt5_use_modules(${PROJECT_NAME} Charts Core)

