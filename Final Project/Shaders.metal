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

    float3 sampleCoord = float3(in.texCoord.x, in.texCoord.y, 0);
    half4 sample = colorMap.sample(colorSampler, sampleCoord.xy);
    float4 color = float4(sample);
    
    return color;
}

struct IntersectionResult {
    bool didIntercept [[accept_intersection]];
    float dist [[distance]];
};

struct Box {
    float3 boxMin;
    float3 boxMax;
};

kernel void interceptCube(acceleration_structure<> primStruct [[buffer(0)]],
                            constant Uniforms & uniforms [[buffer(1)]],
                            texture2d<float, access::write> dstTex [[texture(0)]],
                            texture3d<uint> brickPool [[texture(BrickPoolIndex)]],
                            uint2 tid [[thread_position_in_grid]])
{
    
    if (tid.x < uniforms.tWidth && tid.y < uniforms.tHeight) {
        constexpr sampler colorSampler(filter::linear, coord::pixel, address::clamp_to_border);

        ray r;
        
        float2 pixel = (float2)tid;
        
        float2 uv = (float2)pixel / float2(uniforms.tWidth, uniforms.tHeight);
        
        constant Camera & camera = uniforms.camera;
        
        r.origin = camera.position;
        r.direction = normalize((uv.x - 0.5) * camera.right + (uv.y - 0.5) * camera.up + camera.forward);
        r.max_distance = 10;
        
        intersector<triangle_data> intersector;
        intersection_result<triangle_data> intersection;
        
        intersection = intersector.intersect(r, primStruct);
        
        if( intersection.type == intersection_type::none || intersection.triangle_front_facing == false) {
            dstTex.write(float4(0, 0, 0, 1), tid);
            return;
        }

        float3 xyz = r.origin + r.direction * intersection.distance;
        xyz.x += 0.5;
        xyz.y += 0.5;
        xyz.z += 0.5;
        
        xyz.x *= uniforms.width;
        xyz.y *= uniforms.height;
        xyz.z *= uniforms.depth;
        
        xyz.x /= uniforms.dFactor.x;
        xyz.y /= uniforms.dFactor.y;
        xyz.z /= uniforms.dFactor.z;

        float sample = brickPool.sample(colorSampler, xyz).a;
        float4 sampleColor = {0, 0, 0, 1};
        float3 view = {0};
        float3 normalSample = {0};
        float3 normal = {0};
        float3 n = {0};
        float3 l = {0};
        float3 h = {0};
        
        float4 pos = {0};
        float4 lpos = {0};
        
        float3 ambient = {0.0, 0.0, 0.0};
        float3 diffuse = {0.5, 0.5, 0.5};
        float3 specular = {1, 1, 1};
        float shininess = 2;
        
        if(xyz.x < 0.0){
            xyz.x = -xyz.x;
        } else if(xyz.x > uniforms.maxDim) {
            xyz.x = uniforms.maxDim - (xyz.x - uniforms.maxDim);
        }
        
        if(xyz.y < 0.0){
            xyz.y = -xyz.y;
        } else if(xyz.y > uniforms.maxDim) {
            xyz.y = uniforms.maxDim - (xyz.y - uniforms.maxDim);
        }
        
        if(xyz.z < 0.0){
            xyz.z = -xyz.z;
        } else if(xyz.z > uniforms.maxDim) {
            xyz.z = uniforms.maxDim - (xyz.z - uniforms.maxDim);
        }

        while(xyz.x <= uniforms.maxDim && xyz.y <= uniforms.maxDim && xyz.z <= uniforms.maxDim &&
              xyz.x > 0.0 && xyz.y > 0.0 && xyz.z > 0.0) {

            xyz += r.direction;
            sample = brickPool.sample(colorSampler, xyz).a;
            if(sample >= uniforms.isoValue) {
                pos = uniforms.modelViewMatrix * float4(xyz, 1);
                lpos = uniforms.modelViewMatrix * float4(uniforms.lightPos, 1);
                
                view = normalize( -pos.xyz );
                normalSample = float3(brickPool.sample( colorSampler, xyz ).xyz);
                normal = ( normalSample / 100.0 ) - 1.0;
                n = normalize( uniforms.normalMatrix * float4( normal, 0 ) ).xyz;
                l = normalize( pos.xyz - lpos.xyz );
                h = normalize( view + l );
                
                sampleColor = float4(ambient + diffuse * max( 0.0, dot( n, l ) ) + specular * pow( max( 0.0, dot( n, h ) ), shininess ), 1);
                break;
            }
        }
        
        dstTex.write( sampleColor, tid );
    }
}

float3 calculateDerivatives(uint3 p, uint3 c) {
    float3 pf = float3(p);
    float3 cf = float3(c);
    float dist = distance(pf,cf);
    float distCubed = pow(dist, 3);
    float3 out;
    
    out.x = cf.x - pf.x;
    out.y = cf.y - pf.y;
    out.z = cf.z - pf.z;
    
    return out / distCubed;
}

kernel void calculateGradient(constant Uniforms & uniforms [[buffer(1)]],
                              texture3d<uint, access::read_write> brickPool [[texture(BrickPoolIndex)]],
                              uint3 tid [[thread_position_in_grid]])
{    
    uint3 refVoxel = tid;
    uint3 voxel = tid;

    uint sample = brickPool.read(voxel).a;
    sample = brickPool.read(tid).a;
    
    if(tid.x == 0){
        brickPool.write(uint4(0, 100, 100, sample), tid);
        return;
    }else if (tid.y == 0){
        brickPool.write(uint4(100, 0, 100, sample), tid);
        return;
    }else if (tid.z == 0){
        brickPool.write(uint4(100, 100, 0, sample), tid);
        return;
    }else if (tid.x == uniforms.width){
        brickPool.write(uint4(200, 100, 100, sample), tid);
        return;
    }else if (tid.y == uniforms.height){
        brickPool.write(uint4(100, 200, 100, sample), tid);
        return;
    }else if (tid.z == uniforms.depth){
        brickPool.write(uint4(100, 100, 200, sample), tid);
        return;
    }
    
    uint surroundingValue = 0;
    uint k = 0;
    uint delta = 0;
    uint originalValue = 0;
    uint maxGradientValue = 0; //should this be negative in any case?
    uint3 maxGradientList[9] = {0};
    
    for(uint x = refVoxel.x-1 ; x<=refVoxel.x+1 ; x++){
        for(uint y = refVoxel.y-1 ; y<=refVoxel.y+1 ; y++){
            for(uint z = refVoxel.z-1 ; z<=refVoxel.z+1 ; z++){
                voxel.x = x;
                voxel.y = y;
                voxel.z = z;
                surroundingValue = brickPool.read(voxel).a;
                originalValue = brickPool.read(refVoxel).a;
                delta = abs(surroundingValue-originalValue);
                if (delta != 0){
                    if (delta > maxGradientValue){ //not considered multiple max values
                        maxGradientValue = delta;
                        k = 0;
                        maxGradientList[k] = voxel;
                        k++;
                    }else if (delta == maxGradientValue){
                        maxGradientList[k] = voxel;
                        k++;
                    }
                }else{
                    continue;
                }
            }
        }
    }
    
    float3 maxGradient = {0};
    for(uint i=0 ; i<k ; i++){
        maxGradient += float3(maxGradientList[i]-refVoxel);
    }
    maxGradient = maxGradient / k;
    
    if(maxGradientValue != 0){
        uint3 ret = uint3((normalize(maxGradient)+1) * 100);
        brickPool.write(uint4(ret.x, ret.y, ret.z, sample), tid);
    }
}

kernel void calculateNormal(constant Uniforms & uniforms [[buffer(1)]],
                              texture3d<uint, access::read_write> brickPool [[texture(BrickPoolIndex)]],
                              uint3 tid [[thread_position_in_grid]])
{
    uint3 refVoxel = tid;
    uint3 voxel = tid;

    uint sample = brickPool.read(voxel).a;
    
    if(tid.x == 0){
        brickPool.write(uint4(200, 100, 100, sample), tid);
        return;
    }else if (tid.y == 0){
        brickPool.write(uint4(100, 200, 100, sample), tid);
        return;
    }else if (tid.z == 0){
        brickPool.write(uint4(100, 100, 200, sample), tid);
        return;
    }else if (tid.x == uniforms.width){
        brickPool.write(uint4(0, 100, 100, sample), tid);
        return;
    }else if (tid.y == uniforms.height){
        brickPool.write(uint4(100, 0, 100, sample), tid);
        return;
    }else if (tid.z == uniforms.depth){
        brickPool.write(uint4(100, 100, 0, sample), tid);
        return;
    }
    
    uint surroundingSample = 0;
    uint k = 0;
    uint3 normalList[9] = {0};
    uint delta = 1;
    
    for(uint x = refVoxel.x-delta ; x<=refVoxel.x+delta ; x++){
        for(uint y = refVoxel.y-delta ; y<=refVoxel.y+delta ; y++){
            for(uint z = refVoxel.z-delta ; z<=refVoxel.z+delta ; z++){
                voxel.x = x;
                voxel.y = y;
                voxel.z = z;
                surroundingSample = brickPool.read(voxel).a;
                if (surroundingSample > sample) {
                    normalList[k] = voxel-refVoxel;
                    k++;
                }
            }
        }
    }
    
    float3 normal = {0};
    
    if(k > 0) {
        for(uint i=0 ; i<k ; i++){
            normal += float3(normalList[i]);
        }
    } else {
        normal = float3(voxel - (uint3(uniforms.width, uniforms.height, uniforms.depth)/2));
    }
    
    uint3 ret = uint3((normalize(normal)+1) * 100);
    brickPool.write(uint4(ret.x, ret.y, ret.z, sample), tid);
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
