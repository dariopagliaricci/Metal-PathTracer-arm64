cmake_minimum_required(VERSION 3.24)

project(PathTracerMetal LANGUAGES CXX OBJCXX)

option(STRICT "Enable strict compiler warnings" OFF)
option(M0_TESTS "Enable Milestone 0 golden image tests" OFF)
option(PATH_TRACER_ENABLE_EMBREE "Enable Embree headless backend" OFF)
option(PATH_TRACER_ENABLE_OIDN "Enable OIDN denoiser integration" ON)
option(PT_DEBUG_TOOLS "Enable HWRT/SWRT debug tooling" OFF)
option(PT_MNEE_SWRT_RAYS "Force MNEE rays to use SWRT in HWRT path (debug)" OFF)
option(PT_MNEE_OCCLUSION_PARITY "Compare HWRT vs SWRT MNEE visibility (debug)" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# Enable testing
enable_testing()

if(PATH_TRACER_ENABLE_EMBREE)
    set(EMBREE_TUTORIALS OFF CACHE BOOL "Disable Embree tutorials" FORCE)
    set(EMBREE_ISPC_SUPPORT OFF CACHE BOOL "Disable Embree ISPC support" FORCE)
    set(EMBREE_SYCL_SUPPORT OFF CACHE BOOL "Disable Embree SYCL support" FORCE)
    set(EMBREE_TASKING_SYSTEM "INTERNAL" CACHE STRING "Use Embree internal tasking" FORCE)
    set(EMBREE_STATIC_LIB ON CACHE BOOL "Build Embree as a static library" FORCE)
    add_subdirectory(external/embree EXCLUDE_FROM_ALL)
    find_package(TBB REQUIRED)
endif()

set(PATH_TRACER_OIDN_LIB_DIR "${CMAKE_SOURCE_DIR}/external/oidn/lib")
set(PATH_TRACER_OIDN_LIBRARIES
    "${PATH_TRACER_OIDN_LIB_DIR}/libOpenImageDenoise.2.3.3.dylib"
    "${PATH_TRACER_OIDN_LIB_DIR}/libOpenImageDenoise_core.2.3.3.dylib"
    "${PATH_TRACER_OIDN_LIB_DIR}/libOpenImageDenoise_device_metal.2.3.3.dylib"
    "${PATH_TRACER_OIDN_LIB_DIR}/libtbb.12.12.dylib"
)

if(PATH_TRACER_ENABLE_OIDN)
    set(_missing_oidn_libs "")
    foreach(_oidn_lib IN LISTS PATH_TRACER_OIDN_LIBRARIES)
        if(NOT EXISTS "${_oidn_lib}")
            list(APPEND _missing_oidn_libs "${_oidn_lib}")
        endif()
    endforeach()

    if(_missing_oidn_libs)
        string(REPLACE ";" "\n  - " _missing_oidn_text "${_missing_oidn_libs}")
        message(WARNING
            "OIDN requested but required libraries were not found:\n"
            "  - ${_missing_oidn_text}\n"
            "Building without OIDN denoiser support. To re-enable, place the arm64 OIDN dylibs in ${PATH_TRACER_OIDN_LIB_DIR} "
            "or configure with -DPATH_TRACER_ENABLE_OIDN=OFF."
        )
        set(PATH_TRACER_ENABLE_OIDN OFF)
        set(PATH_TRACER_OIDN_LIBRARIES "")
    else()
        function(path_tracer_ensure_oidn_abi_symlink link_name target_name)
            set(link_path "${PATH_TRACER_OIDN_LIB_DIR}/${link_name}")
            set(target_path "${PATH_TRACER_OIDN_LIB_DIR}/${target_name}")
            if(NOT EXISTS "${target_path}")
                return()
            endif()
            if(EXISTS "${link_path}" OR IS_SYMLINK "${link_path}")
                return()
            endif()

            execute_process(
                COMMAND "${CMAKE_COMMAND}" -E create_symlink "${target_name}" "${link_name}"
                WORKING_DIRECTORY "${PATH_TRACER_OIDN_LIB_DIR}"
                RESULT_VARIABLE symlink_result
                ERROR_VARIABLE symlink_error
            )
            if(NOT symlink_result EQUAL 0)
                message(WARNING
                    "Failed to create OIDN ABI symlink '${link_name}' -> '${target_name}': ${symlink_error}"
                )
            else()
                message(STATUS "Created OIDN ABI symlink: ${link_name} -> ${target_name}")
            endif()
        endfunction()

        # Some OIDN/TBB binaries use ABI install names (e.g. *.2.dylib, *.12.dylib).
        # Ensure these compatibility symlinks exist next to the versioned dylibs.
        path_tracer_ensure_oidn_abi_symlink("libOpenImageDenoise.2.dylib" "libOpenImageDenoise.2.3.3.dylib")
        path_tracer_ensure_oidn_abi_symlink("libtbb.12.dylib" "libtbb.12.12.dylib")
        message(STATUS "OIDN integration enabled")
    endif()
else()
    message(STATUS "OIDN integration disabled by user option")
    set(PATH_TRACER_OIDN_LIBRARIES "")
endif()

set(SHADER_SOURCES
    shaders/common.metal
    shaders/pathtrace.metal
    shaders/mnee.metal
    shaders/display.metal
)

set(MIKKTSPACE_SOURCE external/MikkTSpace/mikktspace.c)
set_source_files_properties(${MIKKTSPACE_SOURCE} PROPERTIES LANGUAGE CXX)

add_executable(PathTracer
    MACOSX_BUNDLE
    src/main.mm
    src/MetalRenderer.mm
    src/renderer/Accumulation.mm
    src/renderer/MetalContext.mm
    src/renderer/Pipelines.mm
    src/renderer/SceneResources.mm
    src/renderer/SceneManager.mm
    src/assets/GltfLoader.mm
    src/assets/TangentGen.mm
    src/renderer/BvhBuilder.mm
    src/renderer/EnvImportanceSampler.mm
    src/renderer/RenderLoop.mm
    src/renderer/UIOverlay.mm
    src/renderer/UniformBuilder.mm
    src/renderer/SceneAccel.mm
    src/renderer/DenoiserContext.mm
    # M0: radiometric change detector
    src/renderer/SettingsUtils.mm
    src/renderer/ImageWriter.mm
    include/MetalRenderer.h
    include/MetalRendererTypes.h
    include/MetalShaderTypes.h
    include/IntersectionProvider.h
    include/renderer/Accumulation.h
    include/renderer/MetalHandles.h
    include/renderer/MetalContext.h
    include/renderer/Pipelines.h
    include/renderer/SceneResources.h
    include/renderer/SceneManager.h
    include/assets/GltfLoader.h
    include/assets/TangentGen.h
    include/renderer/SceneAccel.h
    include/renderer/EnvImportanceSampler.h
    include/renderer/BvhBuilder.h
    include/renderer/PerformanceStats.h
    include/renderer/RenderLoop.h
    include/renderer/RenderSettings.h
    include/renderer/ImageWriter.h
    include/renderer/UIOverlay.h
    include/renderer/UniformBuilder.h
    external/ImGuizmo/ImGuizmo.cpp
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/imgui_demo.cpp
    external/imgui/backends/imgui_impl_metal.mm
    external/imgui/backends/imgui_impl_osx.mm
    ${MIKKTSPACE_SOURCE}
)

# Disable code signing
set_target_properties(PathTracer PROPERTIES
    XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY ""
    XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED "NO"
    XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO"
)

target_include_directories(PathTracer
    PRIVATE
        include
)

target_include_directories(PathTracer
    SYSTEM PRIVATE
        external/imgui
        external/imgui/backends
        external/ImGuizmo
        external/MikkTSpace
        external/tinyobjloader
        external/tinyply
        external/tinybvh
        external/oidn/include
)

target_link_libraries(PathTracer
    PRIVATE
        "-framework Cocoa"
        "-framework QuartzCore"
        "-framework Metal"
        "-framework MetalKit"
        "-framework GameController"
        ${PATH_TRACER_OIDN_LIBRARIES}
)

target_compile_definitions(PathTracer PRIVATE
    PT_ENABLE_OIDN=$<BOOL:${PATH_TRACER_ENABLE_OIDN}>
    PT_DEBUG_TOOLS=$<BOOL:${PT_DEBUG_TOOLS}>
    PT_MNEE_SWRT_RAYS=$<BOOL:${PT_MNEE_SWRT_RAYS}>
    PT_MNEE_OCCLUSION_PARITY=$<BOOL:${PT_MNEE_OCCLUSION_PARITY}>
)

set_source_files_properties(${SHADER_SOURCES} PROPERTIES
    MACOSX_PACKAGE_LOCATION "Resources/shaders"
)

target_sources(PathTracer PRIVATE ${SHADER_SOURCES})

set(ASSET_SOURCE_DIR "${CMAKE_SOURCE_DIR}/assets")
file(GLOB_RECURSE ASSET_FILES CONFIGURE_DEPENDS
    "${ASSET_SOURCE_DIR}/*"
)

set(ASSET_COPY_STAMP "${CMAKE_BINARY_DIR}/PathTracerAssets.stamp")
add_custom_command(
    OUTPUT "${ASSET_COPY_STAMP}"
    COMMAND ${CMAKE_COMMAND} -E rm -rf "$<TARGET_FILE_DIR:PathTracer>/../Resources/assets"
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:PathTracer>/../Resources"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${ASSET_SOURCE_DIR}" "$<TARGET_FILE_DIR:PathTracer>/../Resources/assets"
    COMMAND ${CMAKE_COMMAND} -E touch "${ASSET_COPY_STAMP}"
    DEPENDS ${ASSET_FILES}
    COMMENT "Copying scene assets into PathTracer.app bundle"
)

add_custom_target(PathTracerAssets ALL
    DEPENDS "${ASSET_COPY_STAMP}"
)
add_dependencies(PathTracerAssets PathTracer)

set(HEADLESS_ASSET_COPY_STAMP "${CMAKE_BINARY_DIR}/PathTracerHeadlessAssets.stamp")
add_custom_command(
    OUTPUT "${HEADLESS_ASSET_COPY_STAMP}"
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_BINARY_DIR}/assets" "${CMAKE_BINARY_DIR}/shaders"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/assets" "${CMAKE_BINARY_DIR}/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${ASSET_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/assets"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/shaders" "${CMAKE_BINARY_DIR}/shaders"
    COMMAND ${CMAKE_COMMAND} -E touch "${HEADLESS_ASSET_COPY_STAMP}"
    DEPENDS ${ASSET_FILES} ${SHADER_SOURCES}
    COMMENT "Copying scene assets next to PathTracerHeadless"
)

add_custom_target(PathTracerHeadlessAssets ALL
    DEPENDS "${HEADLESS_ASSET_COPY_STAMP}"
)
add_dependencies(PathTracerHeadlessAssets PathTracerHeadless)

set_source_files_properties(
    src/main.mm
    src/MetalRenderer.mm
    src/renderer/SceneResources.mm
    src/renderer/UIOverlay.mm
    external/imgui/backends/imgui_impl_osx.mm
    src/renderer/SettingsUtils.mm
    external/imgui/backends/imgui_impl_metal.mm
    PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
)

set_source_files_properties(
    external/imgui/backends/imgui_impl_metal.mm
    PROPERTIES
        COMPILE_FLAGS "-fobjc-arc -Wno-unused-variable -Wno-shadow -Wno-implicit-float-conversion"
)

# Headless executable for golden image testing
add_executable(PathTracerHeadless
    src/main_headless.mm
    src/headless/MetalHeadlessRenderer.mm
    src/headless/EmbreeHeadlessRenderer.mm
    src/MetalRenderer.mm
    src/renderer/Accumulation.mm
    src/renderer/MetalContext.mm
    src/renderer/Pipelines.mm
    src/renderer/SceneResources.mm
    src/renderer/SceneManager.mm
    src/assets/GltfLoader.mm
    src/assets/TangentGen.mm
    src/renderer/BvhBuilder.mm
    src/renderer/EnvImportanceSampler.mm
    src/renderer/RenderLoop.mm
    src/renderer/UIOverlay.mm
    src/renderer/UniformBuilder.mm
    src/renderer/SceneAccel.mm
    src/renderer/DenoiserContext.mm
    # M0: radiometric change detector
    src/renderer/SettingsUtils.mm
    src/renderer/ImageWriter.mm
    include/MetalRenderer.h
    include/MetalRendererTypes.h
    include/MetalShaderTypes.h
    include/IntersectionProvider.h
    include/renderer/Accumulation.h
    include/renderer/MetalHandles.h
    include/renderer/MetalContext.h
    include/renderer/Pipelines.h
    include/renderer/SceneResources.h
    include/renderer/SceneManager.h
    include/assets/GltfLoader.h
    include/assets/TangentGen.h
    include/renderer/SceneAccel.h
    include/renderer/EnvImportanceSampler.h
    include/renderer/BvhBuilder.h
    include/renderer/PerformanceStats.h
    include/renderer/RenderLoop.h
    include/renderer/RenderSettings.h
    include/renderer/ImageWriter.h
    include/headless/IHeadlessRenderer.h
    include/headless/MetalHeadlessRenderer.h
    include/headless/EmbreeHeadlessRenderer.h
    include/renderer/UIOverlay.h
    include/renderer/UniformBuilder.h
    external/ImGuizmo/ImGuizmo.cpp
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/imgui_demo.cpp
    external/imgui/backends/imgui_impl_metal.mm
    external/imgui/backends/imgui_impl_osx.mm
    ${MIKKTSPACE_SOURCE}
)

target_include_directories(PathTracerHeadless
    PRIVATE
        include
)

target_include_directories(PathTracerHeadless
    SYSTEM PRIVATE
        external/imgui
        external/imgui/backends
        external/ImGuizmo
        external/MikkTSpace
        external/tinyobjloader
        external/tinyply
        external/tinybvh
        external/oidn/include
)

target_link_libraries(PathTracerHeadless
    PRIVATE
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalKit"
        "-framework QuartzCore"
        "-framework AppKit"
        "-framework GameController"
        ${PATH_TRACER_OIDN_LIBRARIES}
)

target_compile_definitions(PathTracerHeadless PRIVATE
    PT_ENABLE_OIDN=$<BOOL:${PATH_TRACER_ENABLE_OIDN}>
    PT_DEBUG_TOOLS=$<BOOL:${PT_DEBUG_TOOLS}>
    PT_MNEE_SWRT_RAYS=$<BOOL:${PT_MNEE_SWRT_RAYS}>
    PT_MNEE_OCCLUSION_PARITY=$<BOOL:${PT_MNEE_OCCLUSION_PARITY}>
)

if(PATH_TRACER_ENABLE_EMBREE)
    target_compile_definitions(PathTracerHeadless PRIVATE PATH_TRACER_ENABLE_EMBREE=1)
    target_link_libraries(PathTracerHeadless PRIVATE embree)
    target_include_directories(PathTracerHeadless SYSTEM PRIVATE
        $<TARGET_PROPERTY:TBB::tbb,INTERFACE_INCLUDE_DIRECTORIES>
    )

    add_executable(EmbreeSmokeTest
        src/headless/EmbreeSmokeTest.cpp
    )
    target_include_directories(EmbreeSmokeTest PRIVATE include)
    target_link_libraries(EmbreeSmokeTest PRIVATE embree)
        target_compile_definitions(EmbreeSmokeTest PRIVATE
            PT_DEBUG_TOOLS=$<BOOL:${PT_DEBUG_TOOLS}>
            PT_MNEE_SWRT_RAYS=$<BOOL:${PT_MNEE_SWRT_RAYS}>
            PT_MNEE_OCCLUSION_PARITY=$<BOOL:${PT_MNEE_OCCLUSION_PARITY}>
        )
endif()

set_source_files_properties(
    src/main_headless.mm
    src/headless/MetalHeadlessRenderer.mm
    src/headless/EmbreeHeadlessRenderer.mm
    src/MetalRenderer.mm
    src/renderer/SceneResources.mm
    src/renderer/SettingsUtils.mm
    src/renderer/UIOverlay.mm
    PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
)

if(STRICT)
    set(_STRICT_FLAGS -Wall -Wextra -Werror -Wshadow -Wconversion)
    foreach(_target PathTracer PathTracerHeadless)
        if(TARGET ${_target})
            target_compile_options(${_target} PRIVATE ${_STRICT_FLAGS})
        endif()
    endforeach()
endif()

# Copy shaders to build directory for headless executable
add_custom_command(TARGET PathTracerHeadless POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        $<TARGET_FILE_DIR:PathTracerHeadless>/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/shaders/common.metal
        ${CMAKE_SOURCE_DIR}/shaders/pathtrace.metal
        ${CMAKE_SOURCE_DIR}/shaders/mnee.metal
        ${CMAKE_SOURCE_DIR}/shaders/display.metal
        $<TARGET_FILE_DIR:PathTracerHeadless>/shaders/
)

add_custom_command(TARGET PathTracerHeadless POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E rm -rf "$<TARGET_FILE_DIR:PathTracerHeadless>/assets"
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:PathTracerHeadless>/assets"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${ASSET_SOURCE_DIR}"
                                     "$<TARGET_FILE_DIR:PathTracerHeadless>/assets"
    COMMENT "Copying scene assets next to PathTracerHeadless"
)

if(M0_TESTS)
    # Golden image tests
    add_test(
        NAME golden_image_test_generate_after
        COMMAND ${CMAKE_SOURCE_DIR}/tests/tools/golden_test.sh generate after
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_test(
        NAME golden_image_test_compare
        COMMAND ${CMAKE_SOURCE_DIR}/tests/tools/golden_test.sh compare --verbose
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    set_tests_properties(golden_image_test_compare PROPERTIES
        DEPENDS golden_image_test_generate_after
    )
endif()
