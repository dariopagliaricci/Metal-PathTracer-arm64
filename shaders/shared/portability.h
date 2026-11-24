// Shared Metal / C++ dialect (EP8). The headers in shaders/shared are compiled by the Metal shader library and by the
// Embree C++ backend, so their math exists once.
//
// Rules for every header in this folder (the Metal include expander in Pipelines.mm is textual: it inlines every
// line that starts with #include "..." and does not evaluate #if):
// - quoted includes name only files in this folder; C++-only headers come from shared/cpp_prelude.h, which the C++
//   includer includes first and the Metal side never sees;
// - include guards, no #pragma once (the text ends up inside one Metal source string);
// - on Metal every macro below expands to the tokens the code had before it moved here, so the Metal library is
//   unchanged.
#ifndef PT_SHARED_PORTABILITY_H
#define PT_SHARED_PORTABILITY_H

#if !defined(__METAL_VERSION__) && !defined(PT_SHARED_CPP_PRELUDE)
#error "C++ includers of shaders/shared must include shared/cpp_prelude.h first"
#endif

#ifdef __METAL_VERSION__
#define PT_SHARED_BEGIN
#define PT_SHARED_END
// Address space of out-parameters and RNG state.
#define PT_THREAD thread
// Program-scope tables.
#define PT_CONSTANT_TABLE constant
// Read-only material argument: by value on Metal, as the code always passed it.
#define PT_IN(T) const T
// Multi-argument vector constructors: C++ extended vectors have no float3(a, b, c).
#define PT_F2(a, b) float2(a, b)
#define PT_F3(a, b, c) float3(a, b, c)
// Element-wise vector conversion: a C++ cast between float2 and int2 is a bitcast, not a conversion.
#define PT_CONVERT(T, v) T(v)
#define PT_ALL_FINITE3(v) all(isfinite(v))
#else
#define PT_SHARED_BEGIN namespace pt_shared {
#define PT_SHARED_END }
#define PT_THREAD
#define PT_CONSTANT_TABLE inline constexpr
#define PT_IN(T) const T&
#define PT_F2(a, b) simd_make_float2(a, b)
#define PT_F3(a, b, c) simd_make_float3(a, b, c)
#define PT_CONVERT(T, v) __builtin_convertvector(v, T)
#define PT_ALL_FINITE3(v) all_finite(v)
#endif

#endif  // PT_SHARED_PORTABILITY_H
