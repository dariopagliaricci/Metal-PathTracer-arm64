This directory contains the single-header `stb_image.h` decoder used by
`PathTracerImport` for offline texture conversion.

Embree is intentionally not used as the source for this header; the Embree
backend is linked as an installed package via CMake.
