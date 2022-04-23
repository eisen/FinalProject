//
//  ShaderTypes.h
//  Final Project
//
//  Created by Eisen Montalvo on 4/3/22.
//

//
//  Header containing types and enum constants shared between Metal shaders and Swift/ObjC source
//
#ifndef ShaderTypes_h
#define ShaderTypes_h

#ifdef __METAL_VERSION__
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
#define NSInteger metal::int32_t
#else
#import <Foundation/Foundation.h>
#endif

#include <simd/simd.h>

typedef NS_ENUM(NSInteger, BufferIndex)
{
    BufferIndexMeshPositions = 0,
    BufferIndexMeshGenerics  = 1,
    BufferIndexUniforms      = 2
};

typedef NS_ENUM(NSInteger, VertexAttribute)
{
    VertexAttributePosition  = 0,
    VertexAttributeTexcoord  = 1,
};

typedef NS_ENUM(NSInteger, TextureIndex)
{
    TextureIndexColor    = 0,
    BrickPoolIndex       = 1
};

typedef struct {
    vector_float3 position;
    vector_float3 right;
    vector_float3 up;
    vector_float3 forward;
} Camera;

typedef struct
{
    matrix_float4x4 projectionMatrix;
    matrix_float4x4 modelViewMatrix;
    matrix_float4x4 modelMatrix;
    matrix_float4x4 normalMatrix;
    unsigned int width;
    unsigned int height;
    unsigned int depth;
    Camera camera;
    vector_float3 lightPos;
    
    float maxValue;
    float isoValue;
    vector_float3 dFactor;
} Uniforms;

#endif /* ShaderTypes_h */

