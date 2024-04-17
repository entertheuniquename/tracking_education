message("kf[0]:")

cmake_minimum_required(VERSION 3.14)
project(kf VERSION 1.0 LANGUAGES C CXX)

message("kf[1]: установка названия и версии проекта")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_CXX_FLAGS "-std=c++17 -pthread -fPIC")

message("kf[2]: установка флагов")

include_directories("/usr/include/eigen3")

message("kf[3]: включение необходимых дирректорий")

find_package(pybind11 REQUIRED)
if(NOT pybind11_FOUND)
    message("pybind11 is NOT found:")
endif()

find_package(Armadillo REQUIRED)
if(NOT Armadillo_FOUND)
    message("Armadillo is NOT found:")
endif()

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
