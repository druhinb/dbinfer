cmake_minimum_required(VERSION 3.24)

project(dbinfer CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-fno-exceptions -fno-rtti -Wall -Wextra)

option(DBINFER_SANITIZE "Enable ASan/UBSan (Debug builds)" OFF)
if(DBINFER_SANITIZE)
  if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(WARNING "DBINFER_SANITIZE is intended for Debug builds")
  endif()
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

enable_testing()
add_subdirectory(src)
add_subdirectory(tools)
add_subdirectory(tests)
