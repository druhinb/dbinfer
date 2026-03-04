add_library(dbinfer_gguf STATIC
  gguf/gguf.cpp
  gguf/mmap_file.cpp)
target_include_directories(dbinfer_gguf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(dbinfer_tensor STATIC
  tensor/matmul.cpp
  tensor/ops.cpp
  tensor/dequant.cpp)
target_include_directories(dbinfer_tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

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
