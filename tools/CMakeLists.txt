add_executable(gguf_dump_tool gguf_dump_tool.cpp)
target_link_libraries(gguf_dump_tool PRIVATE dbinfer_gguf)

add_executable(gguf_to_dbmf gguf_to_dbmf.cpp)
target_link_libraries(gguf_to_dbmf PRIVATE dbinfer_dbmf)

add_executable(bench bench.cpp)
target_link_libraries(bench PRIVATE dbinfer_model dbinfer_dbmf)

add_executable(grammar_json_gen grammar_json_gen.cpp)
target_link_libraries(grammar_json_gen PRIVATE dbinfer_model dbinfer_grammar dbinfer_sample)
