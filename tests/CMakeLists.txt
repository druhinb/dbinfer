# Phase 0 test harness. The only compiled C++ in this phase.
#
# The golden reload test #includes golden/manifest_generated.hpp, which is
# emitted by tools/dump_reference.py. If the goldens have not been dumped yet
# (fresh checkout before running the dumper), skip the target gracefully so the
# build still succeeds — the plan requires both configs to build regardless.

set(GOLDEN_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/golden/manifest_generated.hpp")

if(EXISTS "${GOLDEN_MANIFEST}")
  add_executable(golden_reload_test golden_reload_test.cpp)
  target_include_directories(golden_reload_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_definitions(golden_reload_test
    PRIVATE DBINFER_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden")
  add_test(NAME golden_reload_test COMMAND golden_reload_test)
else()
  message(STATUS
    "tests/golden/manifest_generated.hpp not found — skipping golden_reload_test. "
    "Run tools/dump_reference.py to generate goldens.")
endif()
