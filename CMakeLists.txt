# 上限卡在 3.26: Humble 还在用 FindPythonInterp, 3.27+ 会配置失败。
cmake_minimum_required(VERSION 3.16...3.26)

# 显式 OLD,避免每次 configure 打 CMP0148 警告。
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()
# 消息生成会出 C 文件,所以语言列表里要有 C。
project(edge_inference_optimizer VERSION 0.1.0 LANGUAGES C CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)

  # Humble 生成的消息头没 include cstdint, GCC 15 会编不过。只给 CXX 强制带上。
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include cstdint>")
endif()

if(NOT CMAKE_BUILD_TYPE)
  # 空的 CMAKE_BUILD_TYPE 会落到 -O0,测延迟没意义。
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# ONNX Runtime: 默认跟 conda 环境走,系统里的 ROS1 不要碰。
if(NOT ONNXRUNTIME_ROOT)
  if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
  elseif(DEFINED ENV{CONDA_PREFIX})
    set(ONNXRUNTIME_ROOT $ENV{CONDA_PREFIX})
  endif()
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime/core/session/onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT}/include
  PATH_SUFFIXES ""
)
find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  HINTS ${ONNXRUNTIME_ROOT}/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR
    "ONNX Runtime not found.\n"
    "  searched root : ${ONNXRUNTIME_ROOT}\n"
    "  include       : ${ONNXRUNTIME_INCLUDE_DIR}\n"
    "  library       : ${ONNXRUNTIME_LIBRARY}\n"
    "Install it with:  conda install -c conda-forge onnxruntime-cpp\n"
    "or point -DONNXRUNTIME_ROOT=/path/to/onnxruntime at a tarball install.")
endif()
message(STATUS "ONNX Runtime include : ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library : ${ONNXRUNTIME_LIBRARY}")

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/StageTiming.msg"
  "msg/LatencyTrace.msg"
  "msg/AudioChunk.msg"
  "msg/VadResult.msg"
  "msg/EmotionResult.msg"
  "msg/AsrResult.msg"
  "msg/LlmReply.msg"
  "msg/ServoCommand.msg"
  DEPENDENCIES std_msgs builtin_interfaces
)

# 核心库不依赖 ROS。OpenCV 给 Whisper STFT 和人脸检测用。
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio objdetect)

add_library(eio_core SHARED
  src/onnx_engine.cpp
  src/latency_probe.cpp
  src/servo_mapper.cpp
  src/whisper_asr.cpp
  src/bpe_tokenizer.cpp
  src/qwen_chat.cpp
  src/piper_tts.cpp
)
target_include_directories(eio_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)
# piper 运行时 dlopen libespeak-ng
target_link_libraries(eio_core PUBLIC ${ONNXRUNTIME_LIBRARY} ${OpenCV_LIBS} ${CMAKE_DL_LIBS})

install(TARGETS eio_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

add_executable(benchmark_cpp src/benchmark_cpp.cpp)
target_link_libraries(benchmark_cpp PRIVATE eio_core)
install(TARGETS benchmark_cpp DESTINATION lib/${PROJECT_NAME})

add_executable(asr_cli src/asr_cli.cpp)
target_link_libraries(asr_cli PRIVATE eio_core)
install(TARGETS asr_cli DESTINATION lib/${PROJECT_NAME})

add_executable(chat_cli src/chat_cli.cpp)
target_link_libraries(chat_cli PRIVATE eio_core)
install(TARGETS chat_cli DESTINATION lib/${PROJECT_NAME})

add_executable(tts_cli src/tts_cli.cpp)
target_link_libraries(tts_cli PRIVATE eio_core)
install(TARGETS tts_cli DESTINATION lib/${PROJECT_NAME})

find_package(cv_bridge REQUIRED)

# PortAudio,跟 ONNX 一样从 conda 前缀找。
find_path(PORTAUDIO_INCLUDE_DIR portaudio.h HINTS ${ONNXRUNTIME_ROOT}/include)
find_library(PORTAUDIO_LIBRARY NAMES portaudio HINTS ${ONNXRUNTIME_ROOT}/lib)
if(NOT PORTAUDIO_INCLUDE_DIR OR NOT PORTAUDIO_LIBRARY)
  message(FATAL_ERROR
    "PortAudio not found (needed for microphone input and speaker output).\n"
    "  conda install -c conda-forge portaudio")
endif()
message(STATUS "PortAudio: ${PORTAUDIO_LIBRARY}")

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

function(eio_add_node target_name source)
  add_executable(${target_name} ${source})
  # 不用 keyword 形式,ament_target_dependencies 内部也是无关键字链接。
  target_link_libraries(${target_name} eio_core "${cpp_typesupport_target}")
  ament_target_dependencies(${target_name}
    rclcpp std_msgs sensor_msgs builtin_interfaces ${ARGN})
  install(TARGETS ${target_name} DESTINATION lib/${PROJECT_NAME})
endfunction()

eio_add_node(emotion_detector_node src/emotion_detector_node.cpp cv_bridge OpenCV)
eio_add_node(camera_node           src/camera_node.cpp           cv_bridge OpenCV)
eio_add_node(servo_mapper_node     src/servo_mapper_node.cpp)
eio_add_node(latency_reporter_node src/latency_reporter_node.cpp)
eio_add_node(audio_capture_node    src/audio_capture_node.cpp)
target_include_directories(audio_capture_node PRIVATE ${PORTAUDIO_INCLUDE_DIR})
target_link_libraries(audio_capture_node ${PORTAUDIO_LIBRARY})
eio_add_node(vad_node              src/vad_node.cpp)
eio_add_node(asr_node              src/asr_node.cpp)
eio_add_node(llm_node              src/llm_node.cpp)
eio_add_node(tts_node              src/tts_node.cpp)

eio_add_node(audio_playback_node   src/audio_playback_node.cpp)
target_include_directories(audio_playback_node PRIVATE ${PORTAUDIO_INCLUDE_DIR})
target_link_libraries(audio_playback_node ${PORTAUDIO_LIBRARY})

install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(PROGRAMS
  scripts/benchmark_python.py
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

ament_export_include_directories(include)
ament_export_libraries(eio_core)

if(BUILD_TESTING)
  # 不用 ament_add_gtest: Humble 自带的 gtest 在 GCC 15 编不过。
  find_package(GTest REQUIRED)
  enable_testing()

  add_executable(test_latency_probe test/test_latency_probe.cpp)
  target_link_libraries(test_latency_probe PRIVATE eio_core GTest::gtest)
  add_test(NAME test_latency_probe COMMAND test_latency_probe)

  add_executable(test_onnx_engine test/test_onnx_engine.cpp)
  target_link_libraries(test_onnx_engine PRIVATE eio_core GTest::gtest)
  add_test(NAME test_onnx_engine COMMAND test_onnx_engine)

  add_executable(test_servo_mapping test/test_servo_mapping.cpp)
  target_link_libraries(test_servo_mapping PRIVATE eio_core GTest::gtest)
  add_test(NAME test_servo_mapping COMMAND test_servo_mapping)

  add_executable(test_whisper_features test/test_whisper_features.cpp)
  target_link_libraries(test_whisper_features PRIVATE eio_core GTest::gtest)
  add_test(NAME test_whisper_features COMMAND test_whisper_features)

  add_executable(test_bpe_tokenizer test/test_bpe_tokenizer.cpp)
  target_link_libraries(test_bpe_tokenizer PRIVATE eio_core GTest::gtest)
  add_test(NAME test_bpe_tokenizer COMMAND test_bpe_tokenizer)
endif()

ament_package()
