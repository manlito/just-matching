cmake_minimum_required(VERSION 3.16...3.31)
project(just_matching VERSION 0.1.0 LANGUAGES CXX)

option(JMT_BUILD_CLI "Build the jmt command line tool" ON)
option(JMT_BUILD_TESTS "Build tests" ON)
option(JMT_BUILD_WASM "Emscripten: build the WebAssembly module, JS wrapper and test page (web/)" ON)
option(JMT_BUILD_PYTHON "Build the Python bindings (bindings/python; needs nanobind)" OFF)
# LightGlue's weights (Apache-2.0) can be built into the library: weights/<model>.jmtw, made by
# tools/export_weights.py --model <model> (47.6 MB each). FlexGlue's never are (DINOv3 License).
option(JMT_EMBED_WEIGHTS "Build the LightGlue weights of JMT_EMBED_MODELS into the library" OFF)
set(JMT_EMBED_MODELS "lightglue-aliked;lightglue-raco-aliked" CACHE STRING "Models to embed when JMT_EMBED_WEIGHTS is ON")
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

# FlexGlue's weights are never compiled in (567 MB, and the backbone is under the DINOv3 License): a path or a
# memory blob is given at run time. LightGlue's can be (JMT_EMBED_WEIGHTS).
set(embed_models "")
set(embed_deps "")
if(JMT_EMBED_WEIGHTS)
  foreach(model IN LISTS JMT_EMBED_MODELS)
    if(NOT model MATCHES "^lightglue-")
      message(FATAL_ERROR "JMT_EMBED_MODELS: only LightGlue models can be built in, not '${model}'")
    endif()
    set(file "${CMAKE_CURRENT_SOURCE_DIR}/weights/${model}.jmtw")
    if(NOT EXISTS "${file}")
      message(FATAL_ERROR "JMT_EMBED_WEIGHTS: ${file} is missing. Make it with "
                          "tools/fetch_lightglue.sh && python tools/export_weights.py --model ${model}")
    endif()
    list(APPEND embed_models "${model}=${file}")
    list(APPEND embed_deps "${file}")
  endforeach()
endif()
set(embed_cpp "${CMAKE_CURRENT_BINARY_DIR}/embedded_weights.cpp")
string(REPLACE ";" "|" embed_models_arg "${embed_models}")  # a list does not survive the command line
add_custom_command(
  OUTPUT "${embed_cpp}"
  COMMAND ${CMAKE_COMMAND} "-DOUT=${embed_cpp}" "-DMODELS=${embed_models_arg}" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake"
  VERBATIM)
# The generated file names the weights files; #embed reads them when it is compiled.
set_source_files_properties("${embed_cpp}" PROPERTIES OBJECT_DEPENDS "${embed_deps}")

add_library(jmt
  src/core/api.cpp
  src/core/weights.cpp
  src/core/checkpoint.cpp
  src/core/ops.cpp
  src/models/flexglue/flexglue.cpp
  src/models/lightglue/lightglue.cpp
  src/backend/cpu/cpu_ops.cpp
  "${embed_cpp}")
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
  set(jmtw_exports jmtw_backend_available jmtw_create jmtw_model jmtw_describe jmtw_describe_keypoints jmtw_features_free
      jmtw_finish jmtw_match jmtw_idx
      jmtw_score jmtw_a_in_b jmtw_b_in_a jmtw_conf_a jmtw_conf_b jmtw_covisibility jmtw_stop_layer jmtw_covisibility_level
      jmtw_level1_matrix jmtw_shortlist jmtw_pairs_idx jmtw_pairs_score jmtw_error jmtw_destroy
      jmtw_convert jmtw_converted jmtw_converted_free jmtw_convert_error)
  list(TRANSFORM jmtw_exports PREPEND "_" OUTPUT_VARIABLE jmtw_exported)
  string(REPLACE ";" "," jmtw_exported "_malloc,_free,${jmtw_exported}")
  target_link_options(jmt_wasm PRIVATE
    -sJSPI
    "-sJSPI_EXPORTS=jmtw_create,jmtw_describe,jmtw_describe_keypoints,jmtw_finish,jmtw_match,jmtw_covisibility_level,jmtw_level1_matrix,jmtw_shortlist"
    "-sEXPORTED_FUNCTIONS=${jmtw_exported}"
    "-sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAP32,HEAPU32,HEAPF32,UTF8ToString"
    -sMODULARIZE
    -sEXPORT_ES6
    -sEXPORT_NAME=createJmt
    -sENVIRONMENT=web,worker
    -sALLOW_MEMORY_GROWTH
    -sMAXIMUM_MEMORY=4GB
    -sSTACK_SIZE=1MB)
  foreach(f jmt.js test.html demo.html demo.js)
    add_custom_command(TARGET jmt_wasm POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "${CMAKE_CURRENT_SOURCE_DIR}/web/${f}" "${CMAKE_CURRENT_BINARY_DIR}/web/${f}")
  endforeach()
  add_custom_command(TARGET jmt_wasm POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory
                     "${CMAKE_CURRENT_SOURCE_DIR}/web/samples" "${CMAKE_CURRENT_BINARY_DIR}/web/samples")
  # LightGlue's weights for the demo, in float16 (24 MB, under the 25 MiB file limit of static hosts):
  # weights/lightglue-aliked-fp16.jmtw, made by `jmt convert aliked_lightglue.pth ... --fp16`.
  set(JMT_DEMO_LIGHTGLUE "${CMAKE_CURRENT_SOURCE_DIR}/weights/lightglue-aliked-fp16.jmtw" CACHE FILEPATH
      "LightGlue weights served next to the demo")
  if(EXISTS "${JMT_DEMO_LIGHTGLUE}")
    add_custom_command(TARGET jmt_wasm POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "${JMT_DEMO_LIGHTGLUE}" "${CMAKE_CURRENT_BINARY_DIR}/web/lightglue-aliked-fp16.jmtw")
  else()
    message(STATUS "LightGlue weights for the demo not found (${JMT_DEMO_LIGHTGLUE}): the demo will offer FlexGlue only")
  endif()
  # The demo takes its keypoints from just-keypoints' module (jkp.js, jkp_wasm.js, jkp_wasm.wasm; raco.jkpw and
  # aliked-n16.jkpw),
  # copied to web/jkp: by default the one tools/fetch_jkp_web.sh puts in third_party/jkp-web, else the <build>/web
  # of a just-keypoints checkout next to this one; or point JMT_JKP_WEB_DIR at another.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/jkp-web/jkp_wasm.wasm")
    set(jmt_jkp_default "${CMAKE_CURRENT_SOURCE_DIR}/third_party/jkp-web")
  else()
    set(jmt_jkp_default "${CMAKE_CURRENT_SOURCE_DIR}/../just-keypoints/build/wasm/web")
  endif()
  set(JMT_JKP_WEB_DIR "${jmt_jkp_default}" CACHE PATH "just-keypoints' built WebAssembly module, for the demo")
  if(EXISTS "${JMT_JKP_WEB_DIR}/jkp_wasm.wasm")
    foreach(f jkp.js jkp_wasm.js jkp_wasm.wasm raco.jkpw aliked-n16.jkpw)
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
  add_executable(lightglue_golden_test tests/lightglue_golden_test.cpp)
  target_include_directories(lightglue_golden_test PRIVATE src)
  target_link_libraries(lightglue_golden_test PRIVATE jmt)
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
  # LightGlue's checkpoints (tools/fetch_lightglue.sh) and, when built in, the embedded copies.
  foreach(model lightglue-aliked lightglue-raco-aliked)
    string(REPLACE "lightglue-" "" pth "${model}")
    string(REPLACE "-" "_" pth "${pth}")
    set(pth "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lightglue-weights/${pth}_lightglue.pth")
    set(jmtw "${CMAKE_CURRENT_SOURCE_DIR}/weights/${model}.jmtw")
    if(EXISTS "${pth}" AND EXISTS "${jmtw}")
      add_test(NAME checkpoint_${model} COMMAND checkpoint_test "${pth}" "${jmtw}")
    endif()
    if(JMT_EMBED_WEIGHTS AND model IN_LIST JMT_EMBED_MODELS)
      add_test(NAME embedded_${model} COMMAND checkpoint_test "embedded:${model}" "${jmtw}")
    endif()
    # jmt convert: the checkpoint to a .jmtw file, which must hold the checkpoint's tensors bit for bit.
    if(TARGET jmt_cli AND EXISTS "${pth}")
      set(converted "${CMAKE_CURRENT_BINARY_DIR}/convert_test_${model}.jmtw")
      add_test(NAME convert_${model}
               COMMAND sh -c "\"$1\" convert \"$2\" \"$3\" && \"$4\" \"$2\" \"$3\"" convert_test
                       $<TARGET_FILE:jmt_cli> "${pth}" "${converted}" $<TARGET_FILE:checkpoint_test>)
    endif()
  endforeach()
  add_executable(ops_bench tests/ops_bench.cpp)  # timings of single operations, not a test
  target_include_directories(ops_bench PRIVATE src)
  target_link_libraries(ops_bench PRIVATE jmt)
  foreach(backend IN LISTS jmt_backends)
    if(NOT backend STREQUAL "cpu")
      string(REPLACE ":" "_" backend_name "${backend}")
      add_test(NAME ops_${backend_name} COMMAND ops_test ${backend})
    endif()
  endforeach()
  # LightGlue: tools/fetch_lightglue.sh, python tools/export_weights.py --model <model>, then
  # tools/dump_golden_lightglue.sh. Every backend of the build, each GPU one in FP32 and FP16.
  foreach(model lightglue-aliked lightglue-raco-aliked)
    set(jmtw "${CMAKE_CURRENT_SOURCE_DIR}/weights/${model}.jmtw")
    file(GLOB lg_golden "${CMAKE_CURRENT_SOURCE_DIR}/build/golden/${model}/*.jmtt")
    if(EXISTS "${jmtw}")
      foreach(golden IN LISTS lg_golden)
        get_filename_component(name "${golden}" NAME_WLE)
        foreach(backend IN LISTS jmt_backends)
          string(REPLACE ":" "_" backend_name "${backend}")
          add_test(NAME ${model}_${backend_name}_${name} COMMAND lightglue_golden_test "${jmtw}" "${golden}" ${backend})
        endforeach()
      endforeach()
    endif()
  endforeach()
  # The C API end to end through the CLI, with the built-in weights and upstream's adaptive defaults: the
  # original's 219 matches after 7 layers on the Berlin pair (keypoints cached by tools/dump_golden_lightglue.py).
  set(berlin_kpts "${CMAKE_CURRENT_SOURCE_DIR}/build/keypoints/berlin1.jpg.aliked.2048.jkpt"
                  "${CMAKE_CURRENT_SOURCE_DIR}/build/keypoints/berlin2.jpg.aliked.2048.jkpt")
  list(GET berlin_kpts 0 berlin_a)
  list(GET berlin_kpts 1 berlin_b)
  if(TARGET jmt_cli AND JMT_EMBED_WEIGHTS AND "lightglue-aliked" IN_LIST JMT_EMBED_MODELS AND EXISTS "${berlin_a}"
     AND EXISTS "${berlin_b}")
    add_test(NAME lightglue_cli_builtin
             COMMAND jmt_cli match --model lightglue-aliked --weights builtin --kpts-a "${berlin_a}" --kpts-b "${berlin_b}")
    set_tests_properties(lightglue_cli_builtin PROPERTIES PASS_REGULAR_EXPRESSION "^219 matches from 2048 and 2048 keypoints, 7 layers")
  endif()
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

if(JMT_BUILD_PYTHON)
  add_subdirectory(bindings/python)
endif()
