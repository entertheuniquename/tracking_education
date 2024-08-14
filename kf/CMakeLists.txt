message("kf[0]:")


cmake_minimum_required(VERSION 3.14)
#CAUSE
project(kf_compare VERSION 1.0 LANGUAGES C CXX)
#project(kf VERSION 1.0 LANGUAGES C CXX)
#~

message("kf[1]: установка названия и версии проекта")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_FLAGS "-std=c++17 -pthread -fPIC")

add_definitions(-DROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

message("kf[2]: установка флагов")

include_directories("/usr/include/eigen3")
include_directories("/usr/local/lib/python3.9/dist-packages")

message("kf[3]: включение необходимых дирректорий")

find_package(pybind11 REQUIRED)
if(NOT pybind11_FOUND)
    message("pybind11 is NOT found:")
endif()

if(pybind11_FOUND)
    message(STATUS "------------------------------------")
    message(STATUS "PYBIND11 is FOUND:")
    message(STATUS "include: ${PYBIND11_INCLUDE_DIRS}")
    message(STATUS "libraries: ${PYBIND11_LIBRARIES}")
    message(STATUS "version: ${PYBIND11_VERSION_STRING}")
    message(STATUS "------------------------------------")
else(pybind11_FOUND)
    message("PYBIND11 is NOT found:")
endif(pybind11_FOUND)

find_package(GTest REQUIRED)

if (GTest_FOUND)
    message(STATUS "---------------------")
    message(STATUS "Google Test is found!")
    message(STATUS "---------------------")
    enable_testing()
else(GTest_FOUND)
    message(STATUS "-------------------------")
    message(STATUS "Google Test is not found!")
    message(STATUS "-------------------------")
endif(GTest_FOUND)

find_package(Boost REQUIRED)

if (Boost_FOUND)
    message(STATUS "---------------------")
    message(STATUS "Boost is found!")
    message(STATUS "---------------------")
    enable_testing()
else(Boost_FOUND)
    message(STATUS "-------------------------")
    message(STATUS "Boost is not found!")
    message(STATUS "-------------------------")
endif(Boost_FOUND)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

if (Python3_FOUND)
    message(STATUS "---------------------")
    message(STATUS "Python3 is found!")
    message(STATUS "---------------------")
    enable_testing()
else(Python3_FOUND)
    message(STATUS "-------------------------")
    message(STATUS "Python3 is not found!")
    message(STATUS "-------------------------")
endif(Python3_FOUND)

include_directories("/usr/include/gtest")
include_directories("/home/ivan/project/3rd_party/armadillo-12.8.2")

find_package(Eigen3 REQUIRED)
if(NOT Eigen3_FOUND)
    message("Eigen3 is NOT found:")
endif()

message("kf[4]: поиск необходимых пакетов")

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test)
set(BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)

message("kf[5]: установка переменных путей")

# Установка дирректории для исполняемого файла
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR})
# Установка дирректории для файлов библиотек
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${BIN_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BIN_DIR})

message("kf[6]: установка дирректорий сборки")

file(GLOB ALL_SRC ${SOURCE_DIR}/* .cpp)
add_library(${PROJECT_NAME} STATIC ${ALL_SRC})

message("kf[7]: создание файла библиотеки")

add_subdirectory(bindpy)
add_subdirectory(test)

message("kf[8]: отработка поддирректорий")

target_link_libraries(${PROJECT_NAME} PRIVATE armadillo)

#target_compile_options(kf PRIVATE "-larmadillo -lblas -llapack")
