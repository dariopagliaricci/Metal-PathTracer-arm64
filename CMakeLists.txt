cmake_minimum_required(VERSION 3.24)

if(APPLE)
    project(PathTracerMetal LANGUAGES CXX OBJCXX)
else()
    project(PathTracerMetal LANGUAGES C CXX)
endif()

option(STRICT "Enable strict compiler warnings" OFF)
option(M0_TESTS "Enable Milestone 0 golden image tests" OFF)
option(PATH_TRACER_ENABLE_EMBREE "Enable Embree headless backend" OFF)
option(PATH_TRACER_ENABLE_ASSIMP "Enable Assimp-based PathTracerImport tool" ON)
option(PATH_TRACER_ENABLE_OIDN "Enable OIDN CPU denoising when vendored OIDN binaries are available" ON)
option(PATH_TRACER_BUILD_EMBREE_SMOKE_ONLY "Build only the standalone Embree smoke target" OFF)
option(PT_DEBUG_TOOLS "Enable HWRT/SWRT debug tooling" OFF)
option(PT_MNEE_SWRT_RAYS "Force MNEE rays to use SWRT in HWRT path (debug)" OFF)
option(PT_MNEE_OCCLUSION_PARITY "Compare HWRT vs SWRT MNEE visibility (debug)" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

set(PATH_TRACER_OIDN_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/oidn/include")
set(PATH_TRACER_OIDN_LIB_DIR "${CMAKE_SOURCE_DIR}/external/oidn/lib")
set(PATH_TRACER_OIDN_LIBS)
set(PATH_TRACER_OIDN_AVAILABLE OFF)
if(PATH_TRACER_ENABLE_OIDN AND NOT PATH_TRACER_BUILD_EMBREE_SMOKE_ONLY)
    # Release checkouts are expected to carry the vendored OIDN/TBB dylibs.
    # Keep detection optional so source-only rebuilds remain reproducible.
    if(EXISTS "${PATH_TRACER_OIDN_INCLUDE_DIR}/OpenImageDenoise/oidn.h")
        find_library(PATH_TRACER_OIDN_LIB
            NAMES OpenImageDenoise libOpenImageDenoise
            PATHS "${PATH_TRACER_OIDN_LIB_DIR}"
            NO_DEFAULT_PATH
        )

        find_library(PATH_TRACER_TBB_LIB
            NAMES tbb libtbb
            PATHS "${PATH_TRACER_OIDN_LIB_DIR}"
            NO_DEFAULT_PATH
        )

        if(PATH_TRACER_OIDN_LIB AND PATH_TRACER_TBB_LIB)
            set(PATH_TRACER_OIDN_AVAILABLE ON)
            set(PATH_TRACER_OIDN_LIBS
                "${PATH_TRACER_OIDN_LIB}"
                "${PATH_TRACER_TBB_LIB}"
            )
        else()
            message(WARNING
                "OIDN was requested but vendored OIDN/TBB binaries were not found in "
                "${PATH_TRACER_OIDN_LIB_DIR}. Building with OIDN disabled; --denoise "
                "will report unavailable."
            )
        endif()
    else()
        message(WARNING
            "OIDN was requested but headers were not found at ${PATH_TRACER_OIDN_INCLUDE_DIR}. "
            "Building with OIDN disabled; --denoise will report unavailable."
        )
    endif()
endif()

# Enable testing
enable_testing()

set(PATH_TRACER_IMPORT_ASSIMP_ENABLED OFF)
if(PATH_TRACER_ENABLE_ASSIMP)
    find_package(assimp CONFIG QUIET)
    if(NOT assimp_FOUND)
        find_package(assimp QUIET)
    endif()

    if(assimp_FOUND)
        set(PATH_TRACER_IMPORT_ASSIMP_ENABLED ON)
    else()
        message(WARNING
            "Assimp was requested but not found. PathTracerImport will build in stub mode. "
            "Install assimp and rerun CMake to enable FBX/OBJ import."
        )
    endif()
endif()

execute_process(
    COMMAND git rev-parse --short=12 HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE PATH_TRACER_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT PATH_TRACER_GIT_HASH)
    set(PATH_TRACER_GIT_HASH "unknown")
endif()

set(PATH_TRACER_IMPORT_VERSION "v3.0.0")

if(PATH_TRACER_ENABLE_EMBREE)
    if(PATH_TRACER_BUILD_EMBREE_SMOKE_ONLY)
        set(PATH_TRACER_ENABLE_ASSIMP OFF CACHE BOOL "Embree smoke-only build disables Assimp targets" FORCE)
    endif()
    find_package(embree 4.4 REQUIRED CONFIG)
    if(TARGET embree)
        set(PATH_TRACER_EMBREE_TARGET embree)
    elseif(TARGET embree::embree)
        set(PATH_TRACER_EMBREE_TARGET embree::embree)
    else()
        message(FATAL_ERROR
            "Embree was found, but its CMake package did not define a usable embree target."
        )
    endif()
elseif(PATH_TRACER_BUILD_EMBREE_SMOKE_ONLY)
    message(FATAL_ERROR "PATH_TRACER_BUILD_EMBREE_SMOKE_ONLY requires PATH_TRACER_ENABLE_EMBREE=ON")
endif()

set(MIKKTSPACE_SOURCE external/MikkTSpace/mikktspace.c)
set_source_files_properties(${MIKKTSPACE_SOURCE} PROPERTIES LANGUAGE CXX)

set(PATH_TRACER_IMGUI_SOURCES
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/imgui_demo.cpp
    external/imgui/backends/imgui_impl_metal.mm
    external/imgui/backends/imgui_impl_osx.mm
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    set_source_files_properties(${PATH_TRACER_IMGUI_SOURCES}
        PROPERTIES
            COMPILE_OPTIONS "-Wno-nontrivial-memcall"
    )
endif()

if(NOT PATH_TRACER_BUILD_EMBREE_SMOKE_ONLY)
    set(SHADER_ENTRY_SOURCES
        shaders/common.metal
        shaders/pathtrace.metal
        shaders/mnee.metal
        shaders/display.metal
    )
    file(GLOB_RECURSE SHADER_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/shaders/*.metal"
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
        src/renderer/StagedSceneLoader.mm
        src/assets/GltfLoader.mm
        src/assets/TangentGen.mm
        src/renderer/BvhBuilder.mm
        src/renderer/EnvImportanceSampler.mm
        src/renderer/RenderLoop.mm
        src/renderer/TransportMemoryAllocator.mm
        src/renderer/RenderPassGraph.cpp
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
        include/renderer/StagedSceneLoader.h
        include/assets/GltfLoader.h
        include/assets/TangentGen.h
        include/renderer/SceneAccel.h
        include/renderer/EnvImportanceSampler.h
        include/renderer/BvhBuilder.h
        include/renderer/PerformanceStats.h
        include/renderer/RenderLoop.h
        include/renderer/TransportMemoryAllocator.h
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
        MACOSX_BUNDLE_GUI_IDENTIFIER "com.dariopagliaricci.PathTracerMetal"
        XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY ""
        XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED "NO"
        XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO"
        XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "com.dariopagliaricci.PathTracerMetal"
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
            "${PATH_TRACER_OIDN_INCLUDE_DIR}"
    )

    target_link_libraries(PathTracer
        PRIVATE
            "-framework Cocoa"
            "-framework QuartzCore"
            "-framework Metal"
            "-framework MetalKit"
            "-framework GameController"
            ${PATH_TRACER_OIDN_LIBS}
    )

    target_compile_definitions(PathTracer PRIVATE
        PATH_TRACER_ENABLE_OIDN=$<BOOL:${PATH_TRACER_OIDN_AVAILABLE}>
        PT_DEBUG_TOOLS=$<BOOL:${PT_DEBUG_TOOLS}>
        PT_MNEE_SWRT_RAYS=$<BOOL:${PT_MNEE_SWRT_RAYS}>
        PT_MNEE_OCCLUSION_PARITY=$<BOOL:${PT_MNEE_OCCLUSION_PARITY}>
    )

    set_source_files_properties(${SHADER_ENTRY_SOURCES} PROPERTIES
        MACOSX_PACKAGE_LOCATION "Resources/shaders"
    )

    target_sources(PathTracer PRIVATE ${SHADER_ENTRY_SOURCES})

    add_custom_target(PathTracerShaderTree
        COMMAND ${CMAKE_COMMAND} -E rm -rf "$<TARGET_FILE_DIR:PathTracer>/../Resources/shaders"
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:PathTracer>/../Resources"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/shaders" "$<TARGET_FILE_DIR:PathTracer>/../Resources/shaders"
        DEPENDS ${SHADER_SOURCES}
        COMMENT "Copying shader tree into PathTracer.app bundle"
    )
    add_dependencies(PathTracer PathTracerShaderTree)

    add_custom_command(TARGET PathTracer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rm -rf "$<TARGET_FILE_DIR:PathTracer>/../Resources/shaders"
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:PathTracer>/../Resources"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/shaders" "$<TARGET_FILE_DIR:PathTracer>/../Resources/shaders"
        COMMENT "Copying shaders into PathTracer.app bundle"
    )

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
        src/renderer/TransportMemoryAllocator.mm
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
        src/renderer/StagedSceneLoader.mm
        src/assets/GltfLoader.mm
        src/assets/TangentGen.mm
        src/renderer/BvhBuilder.mm
        src/renderer/EnvImportanceSampler.mm
        src/renderer/RenderLoop.mm
        src/renderer/TransportMemoryAllocator.mm
        src/renderer/RenderPassGraph.cpp
        src/renderer/UIOverlay.mm
        src/renderer/UniformBuilder.mm
        src/renderer/SceneAccel.mm
        src/renderer/DenoiserContext.mm
        src/renderer/OidnCpuDenoiser.cpp
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
        include/renderer/StagedSceneLoader.h
        include/assets/GltfLoader.h
        include/assets/TangentGen.h
        include/renderer/SceneAccel.h
        include/renderer/EnvImportanceSampler.h
        include/renderer/BvhBuilder.h
        include/renderer/PerformanceStats.h
        include/renderer/RenderLoop.h
        include/renderer/TransportMemoryAllocator.h
        include/renderer/RenderSettings.h
        include/renderer/ImageWriter.h
        include/renderer/OidnCpuDenoiser.h
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

    target_sources(PathTracerHeadless PRIVATE ${SHADER_SOURCES})

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
            "${PATH_TRACER_OIDN_INCLUDE_DIR}"
    )

    target_link_libraries(PathTracerHeadless
        PRIVATE
            "-framework Foundation"
            "-framework Metal"
            "-framework MetalKit"
            "-framework QuartzCore"
            "-framework AppKit"
            "-framework GameController"
            ${PATH_TRACER_OIDN_LIBS}
    )

    target_compile_definitions(PathTracerHeadless PRIVATE
        PATH_TRACER_ENABLE_OIDN=$<BOOL:${PATH_TRACER_OIDN_AVAILABLE}>
        PT_DEBUG_TOOLS=1
        PT_MNEE_SWRT_RAYS=$<BOOL:${PT_MNEE_SWRT_RAYS}>
        PT_MNEE_OCCLUSION_PARITY=$<BOOL:${PT_MNEE_OCCLUSION_PARITY}>
    )
endif()

if(PATH_TRACER_ENABLE_EMBREE)
    add_executable(EmbreeSmokeTest
        src/headless/EmbreeSmokeTest.cpp
    )
    target_include_directories(EmbreeSmokeTest PRIVATE include)
    target_link_libraries(EmbreeSmokeTest PRIVATE ${PATH_TRACER_EMBREE_TARGET})
    target_compile_definitions(EmbreeSmokeTest PRIVATE
        PT_DEBUG_TOOLS=$<BOOL:${PT_DEBUG_TOOLS}>
        PT_MNEE_SWRT_RAYS=$<BOOL:${PT_MNEE_SWRT_RAYS}>
        PT_MNEE_OCCLUSION_PARITY=$<BOOL:${PT_MNEE_OCCLUSION_PARITY}>
    )
    add_test(
        NAME embree_api_smoke
        COMMAND $<TARGET_FILE:EmbreeSmokeTest>
                --width=32
                --height=18
                --output=${CMAKE_BINARY_DIR}/embree_smoke.ppm
    )
    set_tests_properties(embree_api_smoke PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    if(NOT PATH_TRACER_BUILD_EMBREE_SMOKE_ONLY)
        target_compile_definitions(PathTracerHeadless PRIVATE PATH_TRACER_ENABLE_EMBREE=1)
        target_link_libraries(PathTracerHeadless PRIVATE ${PATH_TRACER_EMBREE_TARGET})
    endif()
endif()

if(NOT PATH_TRACER_BUILD_EMBREE_SMOKE_ONLY)
    set_source_files_properties(
        src/main_headless.mm
        src/headless/MetalHeadlessRenderer.mm
        src/headless/EmbreeHeadlessRenderer.mm
        src/MetalRenderer.mm
        src/renderer/SceneResources.mm
        src/renderer/SettingsUtils.mm
        src/renderer/TransportMemoryAllocator.mm
        src/renderer/UIOverlay.mm
        PROPERTIES
            COMPILE_FLAGS "-fobjc-arc"
    )
endif()

if(STRICT)
    set(_STRICT_FLAGS -Wall -Wextra -Werror -Wshadow -Wconversion)
    foreach(_target PathTracer PathTracerHeadless PathTracerImport)
        if(TARGET ${_target})
            target_compile_options(${_target} PRIVATE ${_STRICT_FLAGS})
        endif()
    endforeach()
endif()

if(NOT PATH_TRACER_BUILD_EMBREE_SMOKE_ONLY)
    # Copy shaders to build directory for headless executable
    add_custom_target(PathTracerHeadlessShaders
        COMMAND ${CMAKE_COMMAND} -E make_directory
            $<TARGET_FILE_DIR:PathTracerHeadless>/shaders
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/shaders
            $<TARGET_FILE_DIR:PathTracerHeadless>/shaders
        DEPENDS ${SHADER_SOURCES}
        COMMENT "Copying shaders next to PathTracerHeadless"
    )
    add_dependencies(PathTracerHeadless PathTracerHeadlessShaders)

    add_custom_command(TARGET PathTracerHeadless POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rm -rf "$<TARGET_FILE_DIR:PathTracerHeadless>/assets"
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:PathTracerHeadless>/assets"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${ASSET_SOURCE_DIR}"
                                         "$<TARGET_FILE_DIR:PathTracerHeadless>/assets"
        COMMENT "Copying scene assets next to PathTracerHeadless"
    )

    add_test(
        NAME bistro_full_smoke
        COMMAND $<TARGET_FILE:PathTracerHeadless>
                --scene=bistro_full
                --width=320
                --height=180
                --sppTotal=1
                --enableSoftwareRayTracing=1
                --output=${CMAKE_BINARY_DIR}/bistro_full_smoke.png
    )
    set_tests_properties(bistro_full_smoke PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        SKIP_REGULAR_EXPRESSION "Metal is not supported|Failed to initialize headless Metal context"
    )

    add_executable(PathTracerImport
        src/main_import.mm
        src/import/ImportPipeline.mm
        src/import/TextureConverter.cpp
        include/import/ImportPipeline.h
        include/import/TextureConverter.h
        include/shared/Ktx2Texture.h
    )

    target_include_directories(PathTracerImport
        PRIVATE
            include
    )

    target_include_directories(PathTracerImport
        SYSTEM PRIVATE
            external/stb
    )

    target_link_libraries(PathTracerImport
        PRIVATE
            "-framework Foundation"
            "-framework ImageIO"
            "-framework CoreGraphics"
    )

    target_compile_definitions(PathTracerImport PRIVATE
        PATH_TRACER_ENABLE_ASSIMP=$<BOOL:${PATH_TRACER_IMPORT_ASSIMP_ENABLED}>
        PATH_TRACER_IMPORT_VERSION="${PATH_TRACER_IMPORT_VERSION}"
        PATH_TRACER_GIT_HASH="${PATH_TRACER_GIT_HASH}"
    )

    if(PATH_TRACER_IMPORT_ASSIMP_ENABLED)
        if(TARGET assimp::assimp)
            target_link_libraries(PathTracerImport PRIVATE assimp::assimp)
        elseif(TARGET assimp)
            target_link_libraries(PathTracerImport PRIVATE assimp)
        else()
            target_link_libraries(PathTracerImport PRIVATE ${ASSIMP_LIBRARIES})
            target_include_directories(PathTracerImport SYSTEM PRIVATE ${ASSIMP_INCLUDE_DIRS})
        endif()
    endif()

    add_executable(PathTracerBistroAudit
        src/main_bistro_audit.cpp
        src/import/BistroMaterialAudit.cpp
        src/import/TextureConverter.cpp
        include/import/BistroMaterialAudit.h
        include/import/TextureConverter.h
        include/shared/Ktx2Texture.h
    )

    target_include_directories(PathTracerBistroAudit
        PRIVATE
            include
    )

    target_include_directories(PathTracerBistroAudit
        SYSTEM PRIVATE
            external/stb
    )

    target_compile_definitions(PathTracerBistroAudit PRIVATE
        PATH_TRACER_ENABLE_ASSIMP=$<BOOL:${PATH_TRACER_IMPORT_ASSIMP_ENABLED}>
        PATH_TRACER_IMPORT_VERSION="${PATH_TRACER_IMPORT_VERSION}"
    )

    if(PATH_TRACER_IMPORT_ASSIMP_ENABLED)
        if(TARGET assimp::assimp)
            target_link_libraries(PathTracerBistroAudit PRIVATE assimp::assimp)
        elseif(TARGET assimp)
            target_link_libraries(PathTracerBistroAudit PRIVATE assimp)
        else()
            target_link_libraries(PathTracerBistroAudit PRIVATE ${ASSIMP_LIBRARIES})
            target_include_directories(PathTracerBistroAudit SYSTEM PRIVATE ${ASSIMP_INCLUDE_DIRS})
        endif()
    endif()

    set_source_files_properties(
        src/main_import.mm
        src/import/ImportPipeline.mm
        PROPERTIES
            COMPILE_FLAGS "-fobjc-arc"
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
endif()
