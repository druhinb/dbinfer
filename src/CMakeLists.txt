add_library(dbinfer_gguf STATIC
  gguf/gguf.cpp
  gguf/mmap_file.cpp)
target_include_directories(dbinfer_gguf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Threads REQUIRED)

add_library(dbinfer_tensor STATIC
  tensor/matmul.cpp
  tensor/matmul_neon.cpp
  tensor/cpu.cpp
  tensor/ops.cpp
  tensor/dequant.cpp
  tensor/thread_pool.cpp)
target_include_directories(dbinfer_tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dbinfer_tensor PUBLIC dbinfer_gguf Threads::Threads)

# sdot/i8mm intrinsics need armv8.2 features; guarded behind runtime detection.
set_source_files_properties(tensor/matmul_neon.cpp
  PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod+i8mm")

# Accelerate provides cblas_sgemm for the batched prefill path. absent off Apple,
# where matmul_accel falls back to the scalar reduction.
find_library(ACCELERATE_LIBRARY Accelerate)
if(ACCELERATE_LIBRARY)
  target_link_libraries(dbinfer_tensor PUBLIC ${ACCELERATE_LIBRARY})
  target_compile_definitions(dbinfer_tensor PUBLIC DBINFER_HAVE_ACCELERATE)
endif()

add_library(dbinfer_tokenizer STATIC
  tokenizer/tokenizer.cpp
  tokenizer/bpe.cpp
  tokenizer/unicode.cpp)
target_include_directories(dbinfer_tokenizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dbinfer_tokenizer PUBLIC dbinfer_gguf)

add_library(dbinfer_model STATIC
  model/model.cpp)
target_include_directories(dbinfer_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dbinfer_model PUBLIC dbinfer_gguf dbinfer_tensor dbinfer_tokenizer)

add_library(dbinfer_sample STATIC
  sample/sample.cpp)
target_include_directories(dbinfer_sample PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(dbinfer_args STATIC
  args.cpp)
target_include_directories(dbinfer_args PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dbinfer_args PUBLIC dbinfer_sample)

add_executable(engine main.cpp)
target_link_libraries(engine PRIVATE dbinfer_model dbinfer_args)
set_target_properties(engine PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
