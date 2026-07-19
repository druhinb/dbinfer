# Phase 0 test harness. The only compiled C++ in this phase.
#
# The golden reload test #includes golden/manifest_generated.hpp, which is
# emitted by tools/dump_reference.py. If the goldens have not been dumped yet
# (fresh checkout before running the dumper), skip the target gracefully so the
# build still succeeds — the plan requires both configs to build regardless.

add_executable(gguf_test gguf_test.cpp)
target_link_libraries(gguf_test PRIVATE dbinfer_gguf)
add_test(NAME gguf_test COMMAND gguf_test)

add_executable(tensor_matmul_test tensor_matmul_test.cpp)
target_link_libraries(tensor_matmul_test PRIVATE dbinfer_tensor)
add_test(NAME tensor_matmul_test COMMAND tensor_matmul_test)

add_executable(kv_ring_test kv_ring_test.cpp)
target_link_libraries(kv_ring_test PRIVATE dbinfer_model)
add_test(NAME kv_ring_test COMMAND kv_ring_test)

add_executable(kv_int8_test kv_int8_test.cpp)
target_link_libraries(kv_int8_test PRIVATE dbinfer_model)
add_test(NAME kv_int8_test COMMAND kv_int8_test)

add_executable(dequant_test dequant_test.cpp)
target_link_libraries(dequant_test PRIVATE dbinfer_tensor)
add_test(NAME dequant_test COMMAND dequant_test)

add_executable(int8_dot_test int8_dot_test.cpp)
target_link_libraries(int8_dot_test PRIVATE dbinfer_tensor)
add_test(NAME int8_dot_test COMMAND int8_dot_test)

add_executable(simd_matvec_test simd_matvec_test.cpp)
target_link_libraries(simd_matvec_test PRIVATE dbinfer_tensor)
add_test(NAME simd_matvec_test COMMAND simd_matvec_test)

add_executable(accel_matmul_test accel_matmul_test.cpp)
target_link_libraries(accel_matmul_test PRIVATE dbinfer_tensor)
add_test(NAME accel_matmul_test COMMAND accel_matmul_test)

add_executable(thread_pool_test thread_pool_test.cpp)
target_link_libraries(thread_pool_test PRIVATE dbinfer_tensor)
add_test(NAME thread_pool_test COMMAND thread_pool_test)

add_executable(sample_test sample_test.cpp)
target_link_libraries(sample_test PRIVATE dbinfer_sample)
add_test(NAME sample_test COMMAND sample_test)

add_executable(grammar_test grammar_test.cpp)
target_link_libraries(grammar_test PRIVATE dbinfer_grammar)
add_test(NAME grammar_test COMMAND grammar_test)

add_executable(args_test args_test.cpp)
target_link_libraries(args_test PRIVATE dbinfer_args)
add_test(NAME args_test COMMAND args_test)

set(GOLDEN_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/golden/manifest_generated.hpp")

if(EXISTS "${GOLDEN_MANIFEST}")
  add_executable(golden_reload_test golden_reload_test.cpp)
  target_include_directories(golden_reload_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_definitions(golden_reload_test
    PRIVATE DBINFER_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden")
  add_test(NAME golden_reload_test COMMAND golden_reload_test)

  add_executable(ops_test ops_test.cpp)
  target_link_libraries(ops_test PRIVATE dbinfer_tensor)
  target_compile_definitions(ops_test
    PRIVATE DBINFER_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden")
  add_test(NAME ops_test COMMAND ops_test)
else()
  message(STATUS
    "tests/golden/manifest_generated.hpp not found — skipping golden_reload_test. "
    "Run tools/dump_reference.py to generate goldens.")
endif()

set(TEST_GGUF "${CMAKE_SOURCE_DIR}/tools/llama.cpp/models/qwen2.5-0.5b.gguf")
if(EXISTS "${TEST_GGUF}")
  add_executable(tokenizer_test tokenizer_test.cpp)
  target_include_directories(tokenizer_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(tokenizer_test PRIVATE dbinfer_tokenizer)
  target_compile_definitions(tokenizer_test PRIVATE DBINFER_TEST_GGUF="${TEST_GGUF}")
  add_test(NAME tokenizer_test COMMAND tokenizer_test)

  add_executable(kv_prefix_cache_test kv_prefix_cache_test.cpp)
  target_include_directories(kv_prefix_cache_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(kv_prefix_cache_test PRIVATE dbinfer_model)
  target_compile_definitions(kv_prefix_cache_test PRIVATE DBINFER_TEST_GGUF="${TEST_GGUF}")
  add_test(NAME kv_prefix_cache_test COMMAND kv_prefix_cache_test)

  add_executable(chunked_prefill_test chunked_prefill_test.cpp)
  target_include_directories(chunked_prefill_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(chunked_prefill_test PRIVATE dbinfer_model)
  target_compile_definitions(chunked_prefill_test PRIVATE DBINFER_TEST_GGUF="${TEST_GGUF}")
  add_test(NAME chunked_prefill_test COMMAND chunked_prefill_test)

  if(EXISTS "${GOLDEN_MANIFEST}")
    add_executable(model_test model_test.cpp)
    target_include_directories(model_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(model_test PRIVATE dbinfer_model)
    target_compile_definitions(model_test PRIVATE
      DBINFER_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden"
      DBINFER_TEST_GGUF="${TEST_GGUF}")
    add_test(NAME model_test COMMAND model_test)
  else()
    message(STATUS "goldens not found — skipping model_test.")
  endif()
else()
  message(STATUS "qwen2.5-0.5b.gguf not found — skipping tokenizer_test.")
endif()

set(QUANT_Q8 "${CMAKE_SOURCE_DIR}/build/quant/qwen2.5-0.5b-q8_0.gguf")
if(EXISTS "${QUANT_Q8}" AND EXISTS "${GOLDEN_MANIFEST}")
  add_executable(quant_logits_test quant_logits_test.cpp)
  target_include_directories(quant_logits_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(quant_logits_test PRIVATE dbinfer_model)
  target_compile_definitions(quant_logits_test PRIVATE
    DBINFER_QUANT_Q8_GGUF="${QUANT_Q8}"
    DBINFER_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden")
  add_test(NAME quant_logits_test COMMAND quant_logits_test)
else()
  message(STATUS "q8_0 gguf or goldens not found — skipping quant_logits_test.")
endif()

add_executable(dbmf_convert_test dbmf_convert_test.cpp)
target_include_directories(dbmf_convert_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(dbmf_convert_test PRIVATE dbinfer_model dbinfer_dbmf)
target_compile_definitions(dbmf_convert_test PRIVATE
  DBINFER_DBMF_DIR="${CMAKE_CURRENT_BINARY_DIR}")
if(EXISTS "${TEST_GGUF}")
  target_compile_definitions(dbmf_convert_test PRIVATE DBINFER_TEST_GGUF="${TEST_GGUF}")
endif()
if(EXISTS "${QUANT_Q8}")
  target_compile_definitions(dbmf_convert_test PRIVATE DBINFER_QUANT_Q8="${QUANT_Q8}")
endif()
add_test(NAME dbmf_convert_test COMMAND dbmf_convert_test)

set(SPEC_TARGET "${CMAKE_SOURCE_DIR}/tools/llama.cpp/models/qwen2.5-0.5b.gguf")
set(SPEC_DRAFT "${CMAKE_SOURCE_DIR}/build/quant/qwen2.5-0.5b-q4_0.gguf")
add_executable(speculative_test speculative_test.cpp)
target_include_directories(speculative_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(speculative_test PRIVATE dbinfer_model)
if(EXISTS "${SPEC_TARGET}" AND EXISTS "${SPEC_DRAFT}")
  target_compile_definitions(speculative_test PRIVATE
    DBINFER_SPEC_TARGET="${SPEC_TARGET}"
    DBINFER_SPEC_DRAFT="${SPEC_DRAFT}")
else()
  message(STATUS "spec ggufs not found — speculative_test runs merge-logic only.")
endif()
add_test(NAME speculative_test COMMAND speculative_test)

set(DET_F16 "${CMAKE_SOURCE_DIR}/tools/llama.cpp/models/qwen2.5-0.5b.gguf")
set(DET_Q8 "${CMAKE_SOURCE_DIR}/build/quant/qwen2.5-0.5b-q8_0.gguf")
set(DET_Q4 "${CMAKE_SOURCE_DIR}/build/quant/qwen2.5-0.5b-q4_0.gguf")
set(DET_Q6K "${CMAKE_SOURCE_DIR}/build/quant/qwen2.5-0.5b-q6_k.gguf")
set(DET_Q4K "${CMAKE_SOURCE_DIR}/build/quant/qwen2.5-0.5b-q4_k_m.gguf")
if(EXISTS "${DET_F16}" OR EXISTS "${DET_Q8}" OR EXISTS "${DET_Q4}" OR EXISTS "${DET_Q6K}"
   OR EXISTS "${DET_Q4K}")
  add_executable(matvec_determinism_test matvec_determinism_test.cpp)
  target_include_directories(matvec_determinism_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(matvec_determinism_test PRIVATE dbinfer_model)
  if(EXISTS "${DET_F16}")
    target_compile_definitions(matvec_determinism_test PRIVATE DBINFER_DET_F16="${DET_F16}")
  endif()
  if(EXISTS "${DET_Q8}")
    target_compile_definitions(matvec_determinism_test PRIVATE DBINFER_DET_Q8="${DET_Q8}")
  endif()
  if(EXISTS "${DET_Q4}")
    target_compile_definitions(matvec_determinism_test PRIVATE DBINFER_DET_Q4="${DET_Q4}")
  endif()
  if(EXISTS "${DET_Q6K}")
    target_compile_definitions(matvec_determinism_test PRIVATE DBINFER_DET_Q6K="${DET_Q6K}")
  endif()
  if(EXISTS "${DET_Q4K}")
    target_compile_definitions(matvec_determinism_test PRIVATE DBINFER_DET_Q4K="${DET_Q4K}")
  endif()
  add_test(NAME matvec_determinism_test COMMAND matvec_determinism_test)
else()
  message(STATUS "no qwen ggufs found — skipping matvec_determinism_test.")
endif()
