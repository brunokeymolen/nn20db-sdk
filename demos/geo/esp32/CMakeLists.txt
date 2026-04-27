cmake_minimum_required(VERSION 3.16)

set(IDF_TARGET "esp32p4")

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(NN20DB_COMPONENT_DIR "${REPO_ROOT}/sdk/esp32/current/esp32/component/nn20db")

if(NOT EXISTS "${NN20DB_COMPONENT_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Install the ESP32 SDK first: ${NN20DB_COMPONENT_DIR} is missing")
endif()

list(APPEND EXTRA_COMPONENT_DIRS "${NN20DB_COMPONENT_DIR}")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(geo_esp32)
