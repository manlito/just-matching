cmake_minimum_required(VERSION 3.16...3.31)
project(just_matching VERSION 0.1.0 LANGUAGES CXX)

option(JMT_BUILD_CLI "Build the jmt command line tool" ON)
option(JMT_BUILD_TESTS "Build tests" ON)
set(JMT_METAL "AUTO" CACHE STRING "Metal backend (Apple platforms): AUTO (build it when Metal.framework is found), ON or OFF")
set_property(CACHE JMT_METAL PROPERTY STRINGS AUTO ON OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Metal backend: needs only Metal.framework and Foundation. Shaders are compiled at run time, so no Metal
# toolchain is needed to build.
if(NOT JMT_METAL STREQUAL "OFF")
  if(APPLE)
    find_library(JMT_METAL_FRAMEWORK Metal)
    find_library(JMT_FOUNDATION_FRAMEWORK Foundation)
  endif()
  if(JMT_METAL_FRAMEWORK AND JMT_FOUNDATION_FRAMEWORK)
    enable_language(OBJCXX)
    set(JMT_HAVE_METAL ON)
  elseif(JMT_METAL STREQUAL "ON")
    message(FATAL_ERROR "JMT_METAL=ON but Metal.framework was not found (Apple platforms only)")
  endif()
endif()

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
  target_compile_options(jmt PRIVATE $<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-Wall;-Wextra>)
endif()

set(jmt_backends cpu)
if(JMT_HAVE_METAL)
  set(metal_src "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/metal/kernels.metal")
  set(metal_cpp "${CMAKE_CURRENT_BINARY_DIR}/metal_kernels.cpp")
  add_custom_command(
    OUTPUT "${metal_cpp}"
    COMMAND ${CMAKE_COMMAND} "-DIN=${metal_src}" "-DOUT=${metal_cpp}" -DNAME=kJmtMetalSource
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_text.cmake"
    DEPENDS "${metal_src}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_text.cmake"
    VERBATIM)
  target_sources(jmt PRIVATE src/backend/metal/metal_ops.mm "${metal_cpp}")
  target_compile_definitions(jmt PUBLIC JMT_HAVE_METAL)
  target_compile_options(jmt PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(jmt PRIVATE "${JMT_METAL_FRAMEWORK}" "${JMT_FOUNDATION_FRAMEWORK}")
  list(APPEND jmt_backends metal metal:fp16)
endif()
string(REPLACE ";" ", " jmt_backends_text "${jmt_backends}")
message(STATUS "jmt backends: ${jmt_backends_text}")

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
    foreach(backend IN LISTS jmt_backends)
      string(REPLACE ":" "_" backend_name "${backend}")
      add_test(NAME golden_${backend_name}_${name}
               COMMAND golden_test "${CMAKE_CURRENT_SOURCE_DIR}/weights/flexglue.jmtw" "${golden}" ${backend})
    endforeach()
  endforeach()
endif()
