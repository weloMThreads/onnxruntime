# Copyright (c) Microsoft Corporation. All rights reserved.
# Copyright (c) Moore Threads. All rights reserved.
# Licensed under the MIT License.

  add_definitions(-DUSE_MUSA=1)

  # Enable NHWC ops support if requested
  if (onnxruntime_USE_MUSA_NHWC_OPS)
    add_compile_definitions(ENABLE_MUSA_NHWC_OPS)
  endif()

  # Use the official FindMUSA.cmake to properly find MUSA libraries
  set(onnxruntime_MUSA_HOME "$ENV{MUSA_HOME}" CACHE PATH "Path to MUSA installation directory")
  if (NOT onnxruntime_MUSA_HOME)
    set(onnxruntime_MUSA_HOME "/usr/local/musa")
  endif()
  list(APPEND CMAKE_MODULE_PATH "${onnxruntime_MUSA_HOME}/cmake")
  set(MUSA_TOOLKIT_ROOT_DIR "${onnxruntime_MUSA_HOME}")
  find_package(MUSA REQUIRED)
  list(APPEND MUSA_MCC_FLAGS -std=c++17)

  file(GLOB_RECURSE onnxruntime_providers_musa_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/musa/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/musa/*.cc"
  )

  # Add .mu files for MUSA GPU kernels
  file(GLOB_RECURSE onnxruntime_providers_musa_mu_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/musa/*.mu"
  )
  list(APPEND onnxruntime_providers_musa_mu_srcs
    "${ONNXRUNTIME_ROOT}/core/providers/musa/tensor/gather_impl.mu"
    "${ONNXRUNTIME_ROOT}/core/providers/musa/tensor/gather_elements_impl.mu"
  )
  list(REMOVE_DUPLICATES onnxruntime_providers_musa_mu_srcs)

  # The shared_library files are in a separate list since they use precompiled headers, and the above files have them disabled.
  file(GLOB_RECURSE onnxruntime_providers_musa_shared_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
  )

  # Set .mu files as MUSA source files (automatically recognized by mcc)
  if(onnxruntime_providers_musa_mu_srcs)
    set_source_files_properties(${onnxruntime_providers_musa_mu_srcs}
        PROPERTIES 
        MUSA_SOURCE_PROPERTY_FORMAT OBJ
    )
  endif()

  if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
    file(GLOB_RECURSE onnxruntime_musa_contrib_ops_cc_srcs CONFIGURE_DEPENDS
      "${ONNXRUNTIME_ROOT}/contrib_ops/musa/*.h"
      "${ONNXRUNTIME_ROOT}/contrib_ops/musa/*.cc"
    )
    file(GLOB_RECURSE onnxruntime_musa_contrib_ops_mu_srcs CONFIGURE_DEPENDS
      "${ONNXRUNTIME_ROOT}/contrib_ops/musa/*.mu"
    )
  endif()

  # Set .mu source properties for contrib_ops/musa
  if(onnxruntime_musa_contrib_ops_mu_srcs)
    set_source_files_properties(${onnxruntime_musa_contrib_ops_mu_srcs}
        PROPERTIES
        MUSA_SOURCE_PROPERTY_FORMAT OBJ
    )
  endif()

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_musa_cc_srcs} ${onnxruntime_providers_musa_mu_srcs} ${onnxruntime_providers_musa_shared_srcs})
  set(onnxruntime_providers_musa_src ${onnxruntime_providers_musa_cc_srcs} ${onnxruntime_providers_musa_mu_srcs} ${onnxruntime_providers_musa_shared_srcs})

  # Add contrib_ops/musa sources
  if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
    source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_musa_contrib_ops_cc_srcs} ${onnxruntime_musa_contrib_ops_mu_srcs})
    list(APPEND onnxruntime_providers_musa_src ${onnxruntime_musa_contrib_ops_cc_srcs})
    list(APPEND onnxruntime_providers_musa_src ${onnxruntime_musa_contrib_ops_mu_srcs})
  endif()

  # Use musa_add_library to support .mu files compilation
  musa_add_library(onnxruntime_providers_musa SHARED ${onnxruntime_providers_musa_src})
  onnxruntime_add_include_to_target(onnxruntime_providers_musa onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers Boost::mp11 safeint_interface)

    add_dependencies(onnxruntime_providers_musa onnxruntime_providers_shared ${onnxruntime_EXTERNAL_DEPENDENCIES})

  # Find and use mudnn library for musa::dnn::Tensor
  find_package(PkgConfig QUIET)
  find_library(MUSA_MUDNN_LIBRARY mudnn PATHS "${onnxruntime_MUSA_HOME}/lib" NO_DEFAULT_PATH)
  
  # Find mublas library directly
  find_library(MUSA_MUBLAS_LIBRARY mublas PATHS "${onnxruntime_MUSA_HOME}/lib" NO_DEFAULT_PATH)

    # Link with MUSA libraries including mudnn and mublas (use plain signature to match FindMUSA.cmake)
  target_link_libraries(onnxruntime_providers_musa
    Eigen3::Eigen
    ${MUSA_MUDNN_LIBRARY}
    ${MUSA_MUBLAS_LIBRARY}
    ${ABSEIL_LIBS}
    ${ONNXRUNTIME_PROVIDERS_SHARED})
  target_link_directories(onnxruntime_providers_musa PRIVATE "${onnxruntime_MUSA_HOME}/lib")
  target_include_directories(onnxruntime_providers_musa PRIVATE
    ${ONNXRUNTIME_ROOT}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${MUSA_INCLUDE_DIRS})

  set_target_properties(onnxruntime_providers_musa PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(onnxruntime_providers_musa PROPERTIES FOLDER "ONNXRuntime")

  # Add simplified symbol export configuration (like other providers)
  if(APPLE)
    set_property(TARGET onnxruntime_providers_musa APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${ONNXRUNTIME_ROOT}/core/providers/musa/exported_symbols.lst")
  elseif(UNIX)
    set_property(TARGET onnxruntime_providers_musa APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/musa/version_script.lds -Xlinker --gc-sections")
  elseif(WIN32)
    set_property(TARGET onnxruntime_providers_musa APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${ONNXRUNTIME_ROOT}/core/providers/musa/symbols.def")
  endif()

  install(TARGETS onnxruntime_providers_musa
          ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
