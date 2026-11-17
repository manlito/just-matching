cmake_minimum_required(VERSION 3.16...3.31)
project(just_matching VERSION 0.1.0 LANGUAGES CXX)

option(JMT_BUILD_CLI "Build the jmt command line tool" ON)
option(JMT_BUILD_TESTS "Build tests" ON)
option(JMT_BUILD_WASM "Emscripten: build the WebAssembly module, JS wrapper and test page (web/)" ON)
set(JMT_CUDA "AUTO" CACHE STRING "CUDA backend: AUTO (build it when a CUDA compiler is found), ON or OFF")
set_property(CACHE JMT_CUDA PROPERTY STRINGS AUTO ON OFF)
set(JMT_METAL "AUTO" CACHE STRING "Metal backend (Apple platforms): AUTO (build it when Metal.framework is found), ON or OFF")
set_property(CACHE JMT_METAL PROPERTY STRINGS AUTO ON OFF)
set(JMT_WEBGPU "AUTO" CACHE STRING "WebGPU backend: AUTO (build it when Dawn, or under Emscripten emdawnwebgpu, is found), ON or OFF")
set_property(CACHE JMT_WEBGPU PROPERTY STRINGS AUTO ON OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(EMSCRIPTEN)
  # Native wasm exceptions, as just-keypoints' module: the library reports errors with exceptions, and an
  # app links all its libraries into one module with the same setting.
  add_compile_options(-fwasm-exceptions)
  add_link_options(-fwasm-exceptions)
endif()

# CUDA backend: needs only the CUDA toolkit (nvcc and its runtime library).
if(NOT JMT_CUDA STREQUAL "OFF")
  if(NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    # Toolkit installed but not on PATH (common outside login shells).
    find_program(JMT_NVCC nvcc HINTS ENV CUDA_PATH PATHS /opt/cuda /usr/local/cuda PATH_SUFFIXES bin)
    if(JMT_NVCC)
      set(CMAKE_CUDA_COMPILER "${JMT_NVCC}")
    endif()
  endif()
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(JMT_HAVE_CUDA ON)
  elseif(JMT_CUDA STREQUAL "ON")
    message(FATAL_ERROR "JMT_CUDA=ON but no CUDA compiler was found (set CUDACXX or CMAKE_CUDA_COMPILER)")
  endif()
endif()

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

# WebGPU backend: natively against Dawn's prebuilt release, under Emscripten against emdawnwebgpu from the
# same release. Neither is committed: tools/fetch_webgpu.sh unpacks them under third_party/webgpu, or point
# JMT_DAWN_DIR / JMT_EMDAWNWEBGPU_DIR elsewhere. Kernels are WGSL, compiled at run time.
if(NOT JMT_WEBGPU STREQUAL "OFF")
  if(EMSCRIPTEN)
    set(JMT_EMDAWNWEBGPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/webgpu/emdawnwebgpu/emdawnwebgpu_pkg"
        CACHE PATH "The unpacked emdawnwebgpu_pkg of a Dawn release")
    if(EXISTS "${JMT_EMDAWNWEBGPU_DIR}/emdawnwebgpu.port.py")
      set(JMT_HAVE_WEBGPU ON)
    endif()
  else()
    file(GLOB jmt_dawn_default "${CMAKE_CURRENT_SOURCE_DIR}/third_party/webgpu/dawn/Dawn-*")
    set(JMT_DAWN_DIR "${jmt_dawn_default}" CACHE PATH "The unpacked Dawn release (with lib/cmake/Dawn)")
    # Dawn's Linux release installs into lib64/, macOS's into lib/.
    find_package(Dawn CONFIG QUIET PATHS "${JMT_DAWN_DIR}/lib/cmake/Dawn" "${JMT_DAWN_DIR}/lib64/cmake/Dawn"
                 NO_DEFAULT_PATH)
    if(Dawn_FOUND)
      set(JMT_HAVE_WEBGPU ON)
    endif()
  endif()
  if(NOT JMT_HAVE_WEBGPU AND JMT_WEBGPU STREQUAL "ON")
    message(FATAL_ERROR "JMT_WEBGPU=ON but neither Dawn (JMT_DAWN_DIR) nor, under Emscripten, emdawnwebgpu "
                        "(JMT_EMDAWNWEBGPU_DIR) was found")
  endif()
endif()

# Weights are never compiled in (567 MB, and the backbone is under the DINOv3 License): a path or a memory
# blob is given at run time.
add_library(jmt
  src/core/api.cpp
  src/core/weights.cpp
  src/core/checkpoint.cpp
  src/models/flexglue/flexglue.cpp
  src/backend/cpu/cpu_ops.cpp)
target_include_directories(jmt PUBLIC include PRIVATE src)
find_package(Threads REQUIRED)
target_link_libraries(jmt PRIVATE Threads::Threads)
if(NOT MSVC)
  target_compile_options(jmt PRIVATE $<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-Wall;-Wextra>)
endif()

set(jmt_backends cpu)
if(JMT_HAVE_CUDA)
  target_sources(jmt PRIVATE src/backend/cuda/cuda_ops.cu)
  target_compile_definitions(jmt PUBLIC JMT_HAVE_CUDA)
  set_target_properties(jmt PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  list(APPEND jmt_backends cuda cuda:fp16)
endif()
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
if(JMT_HAVE_WEBGPU)
  set(wgsl_src "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/webgpu/kernels.wgsl")
  set(wgsl_cpp "${CMAKE_CURRENT_BINARY_DIR}/webgpu_kernels.cpp")
  add_custom_command(
    OUTPUT "${wgsl_cpp}"
    COMMAND ${CMAKE_COMMAND} "-DIN=${wgsl_src}" "-DOUT=${wgsl_cpp}" -DNAME=kJmtWgslSource
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_text.cmake"
    DEPENDS "${wgsl_src}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_text.cmake"
    VERBATIM)
  target_sources(jmt PRIVATE src/backend/webgpu/webgpu_ops.cpp "${wgsl_cpp}")
  target_compile_definitions(jmt PUBLIC JMT_HAVE_WEBGPU)
  if(EMSCRIPTEN)
    # PUBLIC: whatever links jmt into a module needs the port's JavaScript library too.
    target_compile_options(jmt PUBLIC "--use-port=${JMT_EMDAWNWEBGPU_DIR}/emdawnwebgpu.port.py")
    target_link_options(jmt PUBLIC "--use-port=${JMT_EMDAWNWEBGPU_DIR}/emdawnwebgpu.port.py")
  else()
    target_link_libraries(jmt PUBLIC dawn::webgpu_dawn)
  endif()
  list(APPEND jmt_backends webgpu webgpu:fp16)
endif()
string(REPLACE ";" ", " jmt_backends_text "${jmt_backends}")
message(STATUS "jmt backends: ${jmt_backends_text}")

# The WebAssembly module (jmt_wasm.js + .wasm) with web/jmt.js and the test page, side by side in
# <build>/web, ready to serve. Single-threaded; waits for the GPU through JSPI. The weights are not copied:
# serve them next to it (tools/web_testdata.py writes the test page's data).
if(EMSCRIPTEN AND JMT_BUILD_WASM)
  add_executable(jmt_wasm web/jmt_wasm.cpp)
  target_include_directories(jmt_wasm PRIVATE src)
  target_link_libraries(jmt_wasm PRIVATE jmt)
  set_target_properties(jmt_wasm PROPERTIES SUFFIX ".js" RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/web")
  set(jmtw_exports jmtw_backend_available jmtw_create jmtw_describe jmtw_features_free jmtw_finish jmtw_match jmtw_idx
      jmtw_score jmtw_a_in_b jmtw_b_in_a jmtw_conf_a jmtw_conf_b jmtw_covisibility jmtw_stop_layer jmtw_covisibility_level
      jmtw_level1_matrix jmtw_shortlist jmtw_pairs_idx jmtw_pairs_score jmtw_error jmtw_destroy)
  list(TRANSFORM jmtw_exports PREPEND "_" OUTPUT_VARIABLE jmtw_exported)
  string(REPLACE ";" "," jmtw_exported "_malloc,_free,${jmtw_exported}")
  target_link_options(jmt_wasm PRIVATE
    -sJSPI
    "-sJSPI_EXPORTS=jmtw_create,jmtw_describe,jmtw_finish,jmtw_match,jmtw_covisibility_level,jmtw_level1_matrix,jmtw_shortlist"
    "-sEXPORTED_FUNCTIONS=${jmtw_exported}"
    "-sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAP32,HEAPU32,HEAPF32,UTF8ToString"
    -sMODULARIZE
    -sEXPORT_ES6
    -sEXPORT_NAME=createJmt
    -sENVIRONMENT=web
    -sALLOW_MEMORY_GROWTH
    -sMAXIMUM_MEMORY=4GB
    -sSTACK_SIZE=1MB)
  foreach(f jmt.js test.html demo.html demo.js)
    add_custom_command(TARGET jmt_wasm POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "${CMAKE_CURRENT_SOURCE_DIR}/web/${f}" "${CMAKE_CURRENT_BINARY_DIR}/web/${f}")
  endforeach()
  add_custom_command(TARGET jmt_wasm POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory
                     "${CMAKE_CURRENT_SOURCE_DIR}/web/samples" "${CMAKE_CURRENT_BINARY_DIR}/web/samples")
  # The demo takes its keypoints from just-keypoints' module: point JMT_JKP_WEB_DIR at the <build>/web of
  # just-keypoints' Emscripten build (jkp.js, jkp_wasm.js, jkp_wasm.wasm, raco.jkpw), copied to web/jkp.
  set(JMT_JKP_WEB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../just-keypoints/build/wasm/web" CACHE PATH
      "just-keypoints' built WebAssembly module, for the demo")
  if(EXISTS "${JMT_JKP_WEB_DIR}/jkp_wasm.wasm")
    foreach(f jkp.js jkp_wasm.js jkp_wasm.wasm raco.jkpw)
      add_custom_command(TARGET jmt_wasm POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                         "${JMT_JKP_WEB_DIR}/${f}" "${CMAKE_CURRENT_BINARY_DIR}/web/jkp/${f}")
    endforeach()
  else()
    message(STATUS "just-keypoints' module not found in JMT_JKP_WEB_DIR (${JMT_JKP_WEB_DIR}): the demo will lack keypoints")
  endif()
endif()

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
  add_executable(batch_test tests/batch_test.cpp)
  target_include_directories(batch_test PRIVATE src)
  target_link_libraries(batch_test PRIVATE jmt)
  add_executable(shortlist_test tests/shortlist_test.cpp)
  target_include_directories(shortlist_test PRIVATE src)
  target_link_libraries(shortlist_test PRIVATE jmt)
  # Each operation of every GPU backend against the CPU backend; needs no weights or golden files.
  add_executable(ops_test tests/ops_test.cpp)
  target_include_directories(ops_test PRIVATE src)
  target_link_libraries(ops_test PRIVATE jmt)
  # flexglue.pth read by the library against the file tools/export_weights.py made from it.
  add_executable(checkpoint_test tests/checkpoint_test.cpp)
  target_include_directories(checkpoint_test PRIVATE src)
  target_link_libraries(checkpoint_test PRIVATE jmt)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/flexglue.pth" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/weights/flexglue.jmtw")
    add_test(NAME checkpoint COMMAND checkpoint_test "${CMAKE_CURRENT_SOURCE_DIR}/third_party/flexglue.pth"
                                                     "${CMAKE_CURRENT_SOURCE_DIR}/weights/flexglue.jmtw")
  endif()
  add_executable(ops_bench tests/ops_bench.cpp)  # timings of single operations, not a test
  target_include_directories(ops_bench PRIVATE src)
  target_link_libraries(ops_bench PRIVATE jmt)
  foreach(backend IN LISTS jmt_backends)
    if(NOT backend STREQUAL "cpu")
      string(REPLACE ":" "_" backend_name "${backend}")
      add_test(NAME ops_${backend_name} COMMAND ops_test ${backend})
    endif()
  endforeach()
  # Golden files and the weights file are generated locally:
  #   python tools/export_weights.py && python tools/dump_golden.py A B
  file(GLOB golden_files "${CMAKE_CURRENT_SOURCE_DIR}/build/golden/*.jmtt")
  foreach(golden IN LISTS golden_files)
    get_filename_component(name "${golden}" NAME_WLE)
    foreach(backend IN LISTS jmt_backends)
      string(REPLACE ":" "_" backend_name "${backend}")
      add_test(NAME golden_${backend_name}_${name}
               COMMAND golden_test "${CMAKE_CURRENT_SOURCE_DIR}/weights/flexglue.jmtw" "${golden}" ${backend})
      add_test(NAME batch_${backend_name}_${name}
               COMMAND batch_test "${CMAKE_CURRENT_SOURCE_DIR}/weights/flexglue.jmtw" "${golden}" ${backend})
      add_test(NAME shortlist_${backend_name}_${name}
               COMMAND shortlist_test "${CMAKE_CURRENT_SOURCE_DIR}/weights/flexglue.jmtw" "${golden}" ${backend})
    endforeach()
  endforeach()
endif()
