add_executable(gguf_dump_tool gguf_dump_tool.cpp)
target_link_libraries(gguf_dump_tool PRIVATE dbinfer_gguf)

add_executable(bench bench.cpp)
target_link_libraries(bench PRIVATE dbinfer_model)
