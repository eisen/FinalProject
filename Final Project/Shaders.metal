//
//  Shaders.metal
//  Final Project
//
//  Created by Eisen Montalvo on 4/3/22.
//

// File for Metal kernel and shader functions

#include <metal_stdlib>
#include <simd/simd.h>

// Including header shared between this Metal shader code and Swift/C code executing Metal API commands
#import "ShaderTypes.h"

using namespace metal;

using namespace metal::raytracing;

typedef struct
{
    float3 position [[attribute(VertexAttributePosition)]];
    float2 texCoord [[attribute(VertexAttributeTexcoord)]];
} Vertex;

typedef struct
{
    float4 position [[position]];
    float2 texCoord;
} ColorInOut;

vertex ColorInOut vertexShader(Vertex in [[stage_in]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]])
{
    ColorInOut out;

    float4 position = float4(in.position, 1.0);
    out.position = uniforms.projectionMatrix * uniforms.modelViewMatrix * position;
    out.texCoord = in.texCoord;

    return out;
}

fragment float4 fragmentShader(ColorInOut in [[stage_in]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                               texture2d<half> colorMap     [[ texture(TextureIndexColor) ]],
                               texture3d<uint> brickPool   [[ texture(BrickPoolIndex) ]])
{
    constexpr sampler colorSampler(mip_filter::linear,
                                   mag_filter::linear);

    float3 sampleCoord = float3(in.texCoord.x / 8.0, in.texCoord.y / 8.0, 0);
    float sample = brickPool.sample(colorSampler, sampleCoord).r / 256.0;
    float4 color;
    color = float4(sample, sample, sample, 1);
    
    return color;
}

struct BoundingBoxResult {
    bool didIntercept [[accept_intersection]];
    float dist [[distance]];
};

struct Box {
    float3 boxMin;
    float3 boxMax;
};

[[intersection(bounding_box, triangle_data, instancing)]]
BoundingBoxResult brickIntersectionFunc(float3 origin                      [[origin]],
                                        float3 direction                   [[direction]],
                                        float minDistance                  [[min_distance]],
                                        float maxDistance                  [[max_distance]],
                                        uint primitiveIndex                [[primitive_id]],
                                        device Box* boundingBoxes          [[buffer(0)]],
                                        ray_data float3& intersectionPoint [[payload]])
{
    float dist = 0.0;
    
    Box bbox = boundingBoxes[primitiveIndex];
    float3 t1 = (bbox.boxMin - origin) / direction;
    float3 t2 = (bbox.boxMax - origin) / direction;
    
    float3 tmin = min(t1.x, t2.x);
    float3 tmax = max(t1.x, t2.x);
    
    bool intersected = min3(tmax.x, tmax.y, tmax.z) > max3(tmin.x, tmin.y, tmin.z);
    
    float3 intPoint;
    if(!intersected) {
        return { false, 0.0f };
    } else {
        intPoint = tmin * direction + origin;
        dist = distance(intPoint, origin);
        if (dist < minDistance || dist > maxDistance) {
            return { false, 0.0f };
        }
        intersectionPoint = intPoint;
    }
    
    return { true, dist };
}

kernel void interceptBricks(instance_acceleration_structure accStruct [[buffer(0)]],
                            primitive_acceleration_structure primStruct [[buffer(1)]],
                            intersection_function_table<triangle_data, instancing> functionTable [[buffer(2)]],
                            constant Uniforms & uniforms [[buffer(3)]],
                            texture2d<float, access::write> dstTex [[texture(0)]],
                            texture3d<uint> brickPool [[texture(BrickPoolIndex)]],
                            uint2 tid [[thread_position_in_grid]])
{
    
    if (tid.x < uniforms.width && tid.y < uniforms.height) {
        constexpr sampler colorSampler(mip_filter::linear,
                                       mag_filter::linear);

        ray r;
        
        float2 pixel = (float2)tid;
        
        float2 uv = (float2)pixel / float2(uniforms.width, uniforms.height);
        
        constant Camera & camera = uniforms.camera;
        
        r.origin = camera.position;
        r.direction = normalize(uv.x * camera.right + uv.y * camera.up + camera.forward);
        r.max_distance = INFINITY;
        
        intersector<triangle_data, instancing> intersector;
        
        intersection_result<triangle_data, instancing> intersection;
        
        float3 intersectionPoint;
        
        intersection = intersector.intersect(r, accStruct, 3, functionTable, intersectionPoint);
        
        if( intersection.type == intersection_type::none) {
            dstTex.write(float4(uv.x, uv.y, 1, 1), tid);
            return;
        }
        
        unsigned int geometryIndex = intersection.geometry_id;
        
        // Convert geometry index to brick pool coordinate
        float3 xyzStart = float3( (geometryIndex % (8 * 9)) / 8.0, ((geometryIndex / 8) % 9) / 8.0, (geometryIndex / 64) / 9.0);
        float3 xyzEnd = float3( ((geometryIndex + 1) % (8 * 9)) / 8.0, (((geometryIndex + 1) / 8) % 9) / 8.0, ((geometryIndex + 1)/ 64) / 9.0);
        
        float3 xyz = xyzStart;
        
        float sample = brickPool.sample(colorSampler, xyz).r;
        
        while(sample < 78 || distance(xyz, xyzEnd) != 0) {
            // March ray
            xyz += r.direction * 0.125; // 1/8
            sample = brickPool.sample(colorSampler, xyz).r;
        }
        
        float sampleColor = sample / 256.0;
        
        dstTex.write(float4(sampleColor, sampleColor, sampleColor, 1), tid);
    }
}

// Screen filling quad in normalized device coordinates.
constant float2 quadVertices[] = {
    float2(-1, -1),
    float2(-1,  1),
    float2( 1,  1),
    float2(-1, -1),
    float2( 1,  1),
    float2( 1, -1)
};

struct CopyVertexOut {
    float4 position [[position]];
    float2 uv;
};

// Simple vertex shader which passes through NDC quad positions.
vertex CopyVertexOut copyVertex(unsigned short vid [[vertex_id]]) {
    float2 position = quadVertices[vid];

    CopyVertexOut out;

    out.position = float4(position, 0, 1);
    out.uv = position * 0.5f + 0.5f;

    return out;
}

// Simple fragment shader which copies a texture and applies a simple tonemapping function.
fragment float4 copyFragment(CopyVertexOut in [[stage_in]],
                             texture2d<float> tex)
{
    constexpr sampler sam(min_filter::nearest, mag_filter::nearest, mip_filter::none);

    float3 color = tex.sample(sam, in.uv).xyz;

    // Apply a very simple tonemapping function to reduce the dynamic range of the
    // input image into a range which the screen can display.
    color = color / (1.0f + color);

    return float4(color, 1.0f);
}
