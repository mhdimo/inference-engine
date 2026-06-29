cmake_minimum_required(VERSION 3.18)
project(InferenceEngine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add compiler flags
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# Find macOS frameworks
if(APPLE)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
endif()

# Core engine library sources
add_library(inference_engine
    engine/core/logger.cpp
    engine/core/sampler.cpp
    engine/core/quantization.cpp
    engine/core/gguf_loader.cpp
    engine/core/tokenizer.cpp
    engine/core/chat_template.cpp
    engine/memory/allocator.cpp
    engine/tensor/tensor.cpp
    engine/backend/cpu/matmul.cpp
    engine/backend/cpu/kernels.cpp
    engine/backend/cpu/deepseek_kernels.cpp
    engine/backend/cpu/quantized_matmul.cpp
    engine/graph/graph.cpp
)

# Compile Metal backend on Apple platforms
if(APPLE)
    target_sources(inference_engine PRIVATE
        engine/backend/metal/metal_backend.mm
    )
    target_link_libraries(inference_engine PUBLIC
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )
    # Enable ARC (Automatic Reference Counting) for the Objective-C++ file
    set_source_files_properties(engine/backend/metal/metal_backend.mm PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )
endif()

# Allow include paths from repo root
target_include_directories(inference_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Unit tests target
add_executable(test_tensor
    tests/test_tensor.cpp
)

target_link_libraries(test_tensor PRIVATE
    inference_engine
)

# LLaMA Block Example target
add_executable(llama_block
    examples/llama_block.cpp
)

target_link_libraries(llama_block PRIVATE
    inference_engine
)

# DeepSeek-V4 Attention Demo target
add_executable(deepseek_v4_demo
    examples/deepseek_v4_demo.cpp
)

target_link_libraries(deepseek_v4_demo PRIVATE
    inference_engine
)

# Text Generator target
add_executable(generate
    examples/generate.cpp
)

target_link_libraries(generate PRIVATE
    inference_engine
)

# Quantization Demo target
add_executable(quant_demo
    examples/quant_demo.cpp
)

target_link_libraries(quant_demo PRIVATE
    inference_engine
)

# GGUF Integration Test target
add_executable(test_gguf
    tests/test_gguf.cpp
)

target_link_libraries(test_gguf PRIVATE
    inference_engine
)

# Tokenizer Integration Test target
add_executable(test_tokenizer
    tests/test_tokenizer.cpp
)

target_link_libraries(test_tokenizer PRIVATE
    inference_engine
)

# Interactive Chat CLI target
add_executable(chat
    examples/chat.cpp
)

target_link_libraries(chat PRIVATE
    inference_engine
)

# SmollM2 Real LLM Chat target
add_executable(smollm2_chat
    examples/smollm2_chat.cpp
)

target_link_libraries(smollm2_chat PRIVATE
    inference_engine
)
