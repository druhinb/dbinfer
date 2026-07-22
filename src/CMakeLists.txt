add_library(dbinfer_gguf STATIC
  gguf/gguf.cpp
  gguf/mmap_file.cpp)
target_include_directories(dbinfer_gguf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(dbinfer_dbmf STATIC
  dbmf/dbmf.cpp
  dbmf/xxhash.cpp
  dbmf/huffman.cpp)
target_include_directories(dbinfer_dbmf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dbinfer_dbmf PUBLIC dbinfer_gguf)

find_package(Threads REQUIRED)

add_library(dbinfer_backend STATIC
  backend/backend.cpp)
target_include_directories(dbinfer_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dbinfer_backend PUBLIC dbinfer_gguf)

# the Metal kernel lives in an Objective-C++ .mm behind a plain C++ header. it
# compiles the mul_mat shader from source at runtime, so no metallib toolchain.
if(DBINFER_METAL)
  find_library(METAL_LIBRARY Metal REQUIRED)
  find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
  target_sources(dbinfer_backend PRIVATE backend/metal_backend.mm)
  set_source_files_properties(backend/metal_backend.mm
    PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
  target_compile_definitions(dbinfer_backend PUBLIC DBINFER_METAL)
  target_link_libraries(dbinfer_backend PUBLIC ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
endif()

add_library(dbinfer_tensor STATIC
  tensor/matmul.cpp
  tensor/matmul_neon.cpp
  tensor/cpu.cpp
  tensor/ops.cpp
  tensor/dequant.cpp
  tensor/thread_pool.cpp)
target_include_directories(dbinfer_tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dbinfer_tensor PUBLIC dbinfer_gguf dbinfer_backend Threads::Threads)

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

add_library(dbinfer_sample STATIC
  sample/sample.cpp)
target_include_directories(dbinfer_sample PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(dbinfer_model STATIC
  model/model.cpp
  model/speculative.cpp)
target_include_directories(dbinfer_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dbinfer_model PUBLIC dbinfer_gguf dbinfer_tensor dbinfer_tokenizer dbinfer_sample)

add_library(dbinfer_grammar STATIC
  grammar/grammar.cpp)
target_include_directories(dbinfer_grammar PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(dbinfer_args STATIC
  args.cpp)
target_include_directories(dbinfer_args PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dbinfer_args PUBLIC dbinfer_sample)

add_executable(engine main.cpp)
target_link_libraries(engine PRIVATE dbinfer_model dbinfer_args dbinfer_grammar dbinfer_dbmf)
set_target_properties(engine PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
