cmake_minimum_required(VERSION 3.16...3.31)
project(just_matching VERSION 0.1.0 LANGUAGES CXX)

option(JMT_BUILD_CLI "Build the jmt command line tool" ON)
option(JMT_BUILD_TESTS "Build tests" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Weights are never compiled in (567 MB, and the backbone is under the DINOv3 License): a path or a memory
# blob is given at run time.
add_library(jmt
  src/core/api.cpp
  src/core/weights.cpp
  src/models/flexglue/flexglue.cpp
  src/backend/cpu/cpu_ops.cpp)
target_include_directories(jmt PUBLIC include PRIVATE src)
find_package(Threads REQUIRED)
target_link_libraries(jmt PRIVATE Threads::Threads)
if(NOT MSVC)
  target_compile_options(jmt PRIVATE -Wall -Wextra)
endif()

message(STATUS "jmt backends: cpu")

if(JMT_BUILD_CLI)
  add_executable(jmt_cli cli/main.cpp)
  set_target_properties(jmt_cli PROPERTIES OUTPUT_NAME jmt)
  target_include_directories(jmt_cli PRIVATE src)
  target_link_libraries(jmt_cli PRIVATE jmt)
endif()

if(JMT_BUILD_TESTS)
  enable_testing()
  add_executable(golden_test tests/golden_test.cpp)
  target_include_directories(golden_test PRIVATE src)
  target_link_libraries(golden_test PRIVATE jmt)
  # Golden files and the weights file are generated locally:
  #   python tools/export_weights.py && python tools/dump_golden.py A B
  file(GLOB golden_files "${CMAKE_CURRENT_SOURCE_DIR}/build/golden/*.jmtt")
  foreach(golden IN LISTS golden_files)
    get_filename_component(name "${golden}" NAME_WLE)
    add_test(NAME golden_cpu_${name}
             COMMAND golden_test "${CMAKE_CURRENT_SOURCE_DIR}/weights/flexglue.jmtw" "${golden}")
  endforeach()
endif()
