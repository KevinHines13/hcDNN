cmake_minimum_required(VERSION 2.8.9)
if (${CMAKE_VERSION} VERSION_GREATER 3.5)
  cmake_policy(SET CMP0046 OLD)
endif()

project(hcDNN)
message(STATUS "Create Project: " ${PROJECT_NAME})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# Cover the include and linkage requirement here
execute_process(COMMAND hcc-config  --cxxflags 
                OUTPUT_VARIABLE HCC_COMPILE_FLAGS)
set(COMPILER_FLAGS "${COMPILER_FLAGS} ${HCC_COMPILE_FLAGS}")

execute_process(COMMAND hcc-config  --ldflags 
                OUTPUT_VARIABLE HCC_LINKER_FLAGS)
set(LINKER_FLAGS "${LINKER_FLAGS} ${HCC_LINKER_FLAGS}")

# Setup normal compilation flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-reorder -pthread")
add_definitions(-Wall -ansi -Wno-deprecated -pthread -std=c++11)

set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR})
include_directories(${COMMON_INCLUDES}) 

# Set source files
file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_SOURCE_DIR} src/*.cc)
message(STATUS "Source files: " ${SOURCES})

message(STATUS ${PATH})
find_program(HCC_FOUND hcc)

# Biuld hcDNN library with HCC
if (HCC_FOUND)

  # Print messge to indicate that CUDA exists
  message(STATUS "HCC exists")

  # Add hcDNN library
  add_library(${PROJECT_NAME} SHARED ${SOURCES})

else()

  message(FATAL_ERROR "HCC not exists")

endif()

# Test
option (enable-test "Compile all hcDNN unit tests" ON)

if (enable-test)
  include(gtest)
  enable_testing()
  set(PROJECT_TEST_NAME ${PROJECT_NAME}_test)
  message(STATUS ${GTEST_INCLUDE_DIRS})
  include_directories(${GTEST_INCLUDE_DIRS})

  # Add source files according to machine setup
  file(GLOB_RECURSE TEST_SRC_FILES RELATIVE 
       ${PROJECT_SOURCE_DIR} 
       test/*.cc
       )
  message(STATUS "Test case source files: " ${TEST_SRC_FILES})

  add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})

  add_dependencies(${PROJECT_TEST_NAME} ${PROJECT_NAME})

  # Link libraries
  target_link_libraries(${PROJECT_TEST_NAME}
      ${GTEST_LIBS_DIR}/libgtest.a
      ${GTEST_LIBS_DIR}/libgtest_main.a
      ${PROJECT_NAME})

  # Add test
  add_test(test1 ${PROJECT_TEST_NAME} "--gtest_color=yes")
endif()

