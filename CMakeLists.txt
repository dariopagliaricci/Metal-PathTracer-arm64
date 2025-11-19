cmake_minimum_required(VERSION 3.24)

project(PathTracerMetal LANGUAGES CXX OBJCXX)

option(STRICT "Enable strict compiler warnings" OFF)
option(M0_TESTS "Enable Milestone 0 golden image tests" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# Enable testing
enable_testing()

set(SHADER_SOURCES
    shaders/common.metal
    shaders/pathtrace.metal
    shaders/display.metal
)

add_executable(PathTracer
    MACOSX_BUNDLE
    src/main.mm
    src/MetalRenderer.mm
    src/renderer/Accumulation.mm
    src/renderer/MetalContext.mm
    src/renderer/Pipelines.mm
    src/renderer/SceneResources.mm
    src/renderer/SceneManager.mm
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
        "${CMAKE_SOURCE_DIR}/external/oidn/lib/libOpenImageDenoise.2.3.3.dylib"
        "${CMAKE_SOURCE_DIR}/external/oidn/lib/libOpenImageDenoise_core.2.3.3.dylib"
        "${CMAKE_SOURCE_DIR}/external/oidn/lib/libOpenImageDenoise_device_metal.2.3.3.dylib"
        "${CMAKE_SOURCE_DIR}/external/oidn/lib/libtbb.12.12.dylib"
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
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_BINARY_DIR}/assets"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/assets"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${ASSET_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/assets"
    COMMAND ${CMAKE_COMMAND} -E touch "${HEADLESS_ASSET_COPY_STAMP}"
    DEPENDS ${ASSET_FILES}
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
    src/MetalRenderer.mm
    src/renderer/Accumulation.mm
    src/renderer/MetalContext.mm
    src/renderer/Pipelines.mm
    src/renderer/SceneResources.mm
    src/renderer/SceneManager.mm
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
        "${CMAKE_SOURCE_DIR}/external/oidn/lib/libOpenImageDenoise.2.3.3.dylib"
        "${CMAKE_SOURCE_DIR}/external/oidn/lib/libOpenImageDenoise_core.2.3.3.dylib"
        "${CMAKE_SOURCE_DIR}/external/oidn/lib/libOpenImageDenoise_device_metal.2.3.3.dylib"
        "${CMAKE_SOURCE_DIR}/external/oidn/lib/libtbb.12.12.dylib"
)

set_source_files_properties(
    src/main_headless.mm
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
