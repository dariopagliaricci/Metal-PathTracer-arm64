inline Ray make_center_primary_ray_for_pixel(constant PathtraceUniforms& uniforms, uint2 pixelCoord) {
    float u = (float(pixelCoord.x) + 0.5f) / float(max(uniforms.width, 1u));
    float v = (float(pixelCoord.y) + 0.5f) / float(max(uniforms.height, 1u));
    v = 1.0f - v;

    Ray ray;
    float3 pixelPosition =
        uniforms.lowerLeftCorner + u * uniforms.horizontal + v * uniforms.vertical;
    ray.origin = uniforms.cameraOrigin;
    ray.direction = pixelPosition - ray.origin;
    return ray;
}
