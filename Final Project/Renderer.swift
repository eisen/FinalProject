//
//  Renderer.swift
//  Final Project
//
//  Created by Eisen Montalvo on 4/3/22.
//

// Our platform independent renderer class

import Metal
import MetalKit
import MetalPerformanceShaders
import simd
import Cocoa

// The 256 byte aligned size of our uniform structure
let alignedUniformsSize = (MemoryLayout<Uniforms>.size + 0xFF) & -0x100

let maxBuffersInFlight = 3

enum RendererError: Error {
    case badVertexDescriptor
}

public enum ScalarSize: Int {
    case BITS_8 = 8
    case BITS_16 = 16
}

class Renderer: NSObject, MTKViewDelegate {

    public let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var dynamicUniformBuffer: MTLBuffer
    var pipelineState: MTLRenderPipelineState
    var rtPipelineState: MTLComputePipelineState?
    var depthState: MTLDepthStencilState
    var colorMap: MTLTexture
    
    var accumulationTargets: [MTLTexture] = []
    var accumulationTargetIdx = 1
    
    var intersectionFunctionTable: MTLIntersectionFunctionTable?
    
    var scalarSize: ScalarSize = .BITS_8
    let fps: FPS = FPS(interval: 0.3)
    var fpsLabel: NSTextField?

    let inFlightSemaphore = DispatchSemaphore(value: maxBuffersInFlight)

    var uniformBufferOffset = 0

    var uniformBufferIndex = 0

    var uniforms: UnsafeMutablePointer<Uniforms>

    var projectionMatrix: matrix_float4x4 = matrix_float4x4()

    var rotation: Float = 0

    var mesh: MTKMesh
    
    var intersector: MPSRayIntersector?
    
    var brickPool: MTLTexture?
    
    var accStructDesc: MTLPrimitiveAccelerationStructureDescriptor?
    
    var instAccStructDesc: MTLInstanceAccelerationStructureDescriptor?
    
    var accStruct: MTLAccelerationStructure?
    
    var accStructs: [MTLAccelerationStructure] = []
    
    var instAccStruct: MTLAccelerationStructure?
    
    var scratchBuffer: MTLBuffer?
    var instScratchBuffer: MTLBuffer?
    var bboxBuffer: MTLBuffer?
    var instBuffer: MTLBuffer?
    
    var size: CGSize?
    
    //var macroCellPool: MTLTexture?
    
    public func SetScalarSize(size: ScalarSize) {
        self.scalarSize = size
    }
    
    public func setVolumeData8(data: Data, dim: vector_int3) {
        let brickPoolDesc = MTLTextureDescriptor.init()
        
        brickPoolDesc.textureType = .type3D
        brickPoolDesc.width = Int(dim.x)
        brickPoolDesc.height = Int(dim.y)
        brickPoolDesc.depth = Int(dim.z)
        brickPoolDesc.pixelFormat = .r8Uint
        
        self.brickPool = self.device.makeTexture(descriptor: brickPoolDesc)!
        self.brickPool?.label = "Brick Pool UInt8"
        
        let region = MTLRegionMake3D(0, 0, 0, Int(dim.x), Int(dim.y), Int(dim.z))
        data.withUnsafeBytes {(bytes: UnsafePointer<UInt8>)->Void in
            self.brickPool!.replace(region: region, mipmapLevel: 0, slice: 0, withBytes: bytes, bytesPerRow: Int(dim.x), bytesPerImage: Int(dim.x) * Int(dim.y))
        }
    }
    
    public func setVolumeData16(data: Data, dim: vector_int3) {
        let brickPoolDesc = MTLTextureDescriptor.init()
        
        brickPoolDesc.textureType = .type3D
        brickPoolDesc.width = Int(dim.x)
        brickPoolDesc.height = Int(dim.y)
        brickPoolDesc.depth = Int(dim.z)
        brickPoolDesc.pixelFormat = .r16Uint
        
        self.brickPool = self.device.makeTexture(descriptor: brickPoolDesc)!
        self.brickPool?.label = "Brick Pool UInt16"
        
        let region = MTLRegionMake3D(0, 0, 0, Int(dim.x), Int(dim.y), Int(dim.z))
        data.withUnsafeBytes {(bytes: UnsafePointer<UInt16>)->Void in
            self.brickPool!.replace(region: region, mipmapLevel: 0, slice: 0, withBytes: bytes, bytesPerRow: Int(dim.x), bytesPerImage: Int(dim.x) * Int(dim.y))
        }
    }
    
    public func setFPSlabel(label: NSTextField) {
        self.fpsLabel = label
    }
    
    public func createAccelerationDesc() {
        var geomDescs: [MTLAccelerationStructureTriangleGeometryDescriptor] = []
        self.accStructDesc = MTLPrimitiveAccelerationStructureDescriptor()
        
        for submesh in mesh.submeshes {
            let cubeDesc = MTLAccelerationStructureTriangleGeometryDescriptor()
            let buffer = mesh.vertexBuffers[0]
            cubeDesc.indexType = submesh.indexType
            cubeDesc.indexBuffer = submesh.indexBuffer.buffer
            cubeDesc.indexBufferOffset = submesh.indexBuffer.offset
            cubeDesc.triangleCount = submesh.indexCount / 3 // Convert to triangles
            cubeDesc.vertexBuffer = buffer.buffer
            cubeDesc.vertexBufferOffset = buffer.offset

            geomDescs.append(cubeDesc)
        }
        
        self.accStructDesc?.geometryDescriptors = geomDescs
        self.accStructDesc?.usage = .refit
        
        let sizes = self.device.accelerationStructureSizes(descriptor: accStructDesc!)
        self.accStruct = self.device.makeAccelerationStructure(size: sizes.accelerationStructureSize)
        self.accStruct?.label = "Primitive Acceleration Structure"
        self.scratchBuffer = self.device.makeBuffer(length: sizes.buildScratchBufferSize, options: .storageModePrivate)
        self.scratchBuffer?.label = "Primitive Scratch Buffer"
        
        self.accStructs.append(self.accStruct!)
        
        let commandBuffer = commandQueue.makeCommandBuffer()
        
        let accStructEncoder = commandBuffer!.makeAccelerationStructureCommandEncoder()!
        
        accStructEncoder.build(accelerationStructure: self.accStruct!, descriptor: self.accStructDesc!, scratchBuffer: self.scratchBuffer!, scratchBufferOffset: 0)
        
        accStructEncoder.endEncoding()
        
        commandBuffer?.commit()
        
        commandBuffer?.waitUntilCompleted()
    }

    init?(metalKitView: MTKView) {
        self.device = metalKitView.device!
        self.commandQueue = self.device.makeCommandQueue()!

        let uniformBufferSize = alignedUniformsSize * maxBuffersInFlight

        self.dynamicUniformBuffer = self.device.makeBuffer(length:uniformBufferSize,
                                                           options:[MTLResourceOptions.storageModeShared])!

        self.dynamicUniformBuffer.label = "Uniforms Buffer"

        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents()).bindMemory(to:Uniforms.self, capacity:1)

        metalKitView.depthStencilPixelFormat = MTLPixelFormat.depth32Float_stencil8
        metalKitView.colorPixelFormat = MTLPixelFormat.bgra8Unorm_srgb
        metalKitView.sampleCount = 1

        let mtlVertexDescriptor = Renderer.buildMetalVertexDescriptor()

        do {
            pipelineState = try Renderer.buildRenderPipelineWithDevice(device: device,
                                                                       metalKitView: metalKitView,
                                                                       mtlVertexDescriptor: mtlVertexDescriptor)
        } catch {
            print("Unable to compile render pipeline state.  Error info: \(error)")
            return nil
        }

        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.less
        depthStateDescriptor.isDepthWriteEnabled = true
        
        self.depthState = device.makeDepthStencilState(descriptor:depthStateDescriptor)!

        do {
            mesh = try Renderer.buildMesh(device: device, mtlVertexDescriptor: mtlVertexDescriptor)
        } catch {
            print("Unable to build MetalKit Mesh. Error info: \(error)")
            return nil
        }

        do {
            colorMap = try Renderer.loadTexture(device: device, textureName: "ColorMap")
        } catch {
            print("Unable to load texture. Error info: \(error)")
            return nil
        }

        super.init()
        
        self.updateGameState()
        
        self.createAccelerationDesc()
        
        do {
            rtPipelineState = try self.buildComputePipelineWithDevice(device: device,
                                                                       metalKitView: metalKitView,
                                                                       mtlVertexDescriptor: mtlVertexDescriptor)
        } catch {
            print("Unable to compile render pipeline state.  Error info: \(error)")
            return nil
        }

    }

    class func buildMetalVertexDescriptor() -> MTLVertexDescriptor {
        // Create a Metal vertex descriptor specifying how vertices will by laid out for input into our render
        //   pipeline and how we'll layout our Model IO vertices

        let mtlVertexDescriptor = MTLVertexDescriptor()

        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].format = MTLVertexFormat.float3
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].bufferIndex = BufferIndex.meshPositions.rawValue

        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stride = 12
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepFunction = MTLVertexStepFunction.perVertex

        return mtlVertexDescriptor
    }

    class func buildRenderPipelineWithDevice(device: MTLDevice,
                                             metalKitView: MTKView,
                                             mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTLRenderPipelineState {
        /// Build a render state pipeline object

        let library = device.makeDefaultLibrary()
        
        let vertexFunction = library?.makeFunction(name: "copyVertex")
        let fragmentFunction = library?.makeFunction(name: "copyFragment")

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "RenderPipeline"
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction

        pipelineDescriptor.colorAttachments[0].pixelFormat = metalKitView.colorPixelFormat
        pipelineDescriptor.depthAttachmentPixelFormat = metalKitView.depthStencilPixelFormat
        pipelineDescriptor.stencilAttachmentPixelFormat = metalKitView.depthStencilPixelFormat

        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }
    
    func buildComputePipelineWithDevice(device: MTLDevice,
                                             metalKitView: MTKView,
                                             mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTLComputePipelineState {
        let library = device.makeDefaultLibrary()!

        let computePipelineDesc = MTLComputePipelineDescriptor()
        computePipelineDesc.computeFunction = library.makeFunction(name: "interceptBricks")!
        
        let computePipelineState = try device.makeComputePipelineState(descriptor: computePipelineDesc, options: .bufferTypeInfo, reflection: nil)
        
        return computePipelineState
    }

    class func buildMesh(device: MTLDevice,
                         mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTKMesh {
        /// Create and condition mesh data to feed into a pipeline using the given vertex descriptor

        let metalAllocator = MTKMeshBufferAllocator(device: device)

        let mdlMesh = MDLMesh.newBox(withDimensions: SIMD3<Float>(1, 1, 1),
                                     segments: SIMD3<UInt32>(1, 1, 1),
                                     geometryType: MDLGeometryType.triangles,
                                     inwardNormals:false,
                                     allocator: metalAllocator)

        let mdlVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(mtlVertexDescriptor)

        guard let attributes = mdlVertexDescriptor.attributes as? [MDLVertexAttribute] else {
            throw RendererError.badVertexDescriptor
        }
        attributes[VertexAttribute.position.rawValue].name = MDLVertexAttributePosition
        attributes[VertexAttribute.texcoord.rawValue].name = MDLVertexAttributeTextureCoordinate

        mdlMesh.vertexDescriptor = mdlVertexDescriptor

        return try MTKMesh(mesh:mdlMesh, device:device)
    }

    class func loadTexture(device: MTLDevice,
                           textureName: String) throws -> MTLTexture {
        /// Load texture data with optimal parameters for sampling

        let textureLoader = MTKTextureLoader(device: device)

        let textureLoaderOptions = [
            MTKTextureLoader.Option.textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            MTKTextureLoader.Option.textureStorageMode: NSNumber(value: MTLStorageMode.`private`.rawValue)
        ]

        return try textureLoader.newTexture(name: textureName,
                                            scaleFactor: 1.0,
                                            bundle: nil,
                                            options: textureLoaderOptions)

    }

    private func updateDynamicBufferState() {
        /// Update the state of our uniform buffers before rendering

        uniformBufferIndex = (uniformBufferIndex + 1) % maxBuffersInFlight

        uniformBufferOffset = alignedUniformsSize * uniformBufferIndex

        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents() + uniformBufferOffset).bindMemory(to:Uniforms.self, capacity:1)
    }

    private func updateGameState() {
        /// Update any game state before rendering

        uniforms[0].projectionMatrix = projectionMatrix

        let rotationAxis = SIMD3<Float>(1, 1, 0)
        let modelMatrix = matrix4x4_rotation(radians: rotation, axis: rotationAxis)
        let viewMatrix = matrix4x4_translation(0.0, 2.0, -2.0)
        
        uniforms[0].modelMatrix = modelMatrix
        uniforms[0].modelViewMatrix = simd_mul(viewMatrix, modelMatrix)
        
        if(self.size != nil) {
            uniforms[0].width = UInt32(self.size!.width)
            uniforms[0].height = UInt32(self.size!.height)
        }
        uniforms[0].camera.position = vector_float3(0, 2, -2)
        uniforms[0].camera.forward = vector_float3(0, -0.70710678118, 0.70710678118)
        uniforms[0].camera.up = vector_float3(0, 0.70710678118, 0.70710678118)
        uniforms[0].camera.right = vector_float3(1, 0, 0)
    }
    
//    private func generateWorkingSet() {
//        let c: Double = 1
//        switch scalarSize {
//        case .BITS_8:
//            if self.octreeUInt8 != nil {
//                self.bricks8Cut = (self.octreeUInt8?.findNodeCut(cameraPos: vector_double3(0, 0, -2), c: c))!
//            }
//        case .BITS_16:
//            if self.octreeUInt16 != nil {
//                self.bricks16Cut = (self.octreeUInt16?.findNodeCut(cameraPos: vector_double3(0, 0, -2), c: c))!
//            }
//        }
//    }
    
//    private func transferWorkingSet() {
//        switch scalarSize {
//        case .BITS_8:
//            for (idx, brick) in bricks8Cut.enumerated() {
//                let x = idx % (8 * 9)
//                let y = (idx / 8) % 9
//                let z = idx / 64
//                let region = MTLRegionMake3D(x, y, z, 128, 128, 128)
//                self.brickPool!.replace(region: region, mipmapLevel: 0, slice: 0, withBytes: brick.scalars, bytesPerRow: 128, bytesPerImage: 128 * 128)
//            }
//
//       case .BITS_16:
//            print("TBD")
//        }
//    }
    
    func draw(in view: MTKView) {
        /// Per frame updates hare

        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)
        
        if let commandBuffer = commandQueue.makeCommandBuffer() {
            let semaphore = inFlightSemaphore
            commandBuffer.addCompletedHandler { (_ commandBuffer)-> Swift.Void in
                semaphore.signal()
                let fpsa: Float = self.fps.getFPS(time: commandBuffer.gpuEndTime)
                if fpsa != 0 {
                    DispatchQueue.main.async {
                        self.fpsLabel!.stringValue = String.init(format: "%0.1f FPS", fpsa)
                    }
                }
            }
            
            self.updateDynamicBufferState()
            
            self.updateGameState()
            
            //self.generateWorkingSet()
            if(self.brickPool != nil) {
                //self.transferWorkingSet()
            
                let width = Int(self.size!.width)
                let height = Int(self.size!.height)
                
                let threadsPerThreadgroup = MTLSizeMake(8, 8, 1)
                let threadgroups = MTLSizeMake((width  + threadsPerThreadgroup.width  - 1) / threadsPerThreadgroup.width,
                                                   (height + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
                                                   1);
                
                let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
                
                computeEncoder.setAccelerationStructure(self.accStruct, bufferIndex: 0)
                
                computeEncoder.setBuffer(dynamicUniformBuffer, offset: 0, index: 1)
                
                computeEncoder.setComputePipelineState(rtPipelineState!)
                
                computeEncoder.setTexture(accumulationTargets[accumulationTargetIdx], index: 0)
                computeEncoder.setTexture(self.brickPool, index: TextureIndex.BrickPoolIndex.rawValue)
                computeEncoder.useResource(self.accStruct!, usage: .read)
                
                computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
                
                computeEncoder.endEncoding()
            }
            
            /// Delay getting the currentRenderPassDescriptor until we absolutely need it to avoid
            ///   holding onto the drawable and blocking the display pipeline any longer than necessary
            let renderPassDescriptor = view.currentRenderPassDescriptor

            if let renderPassDescriptor = renderPassDescriptor {

                /// Final pass rendering code here
                if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
                    renderEncoder.label = "Primary Render Encoder"

                    renderEncoder.pushDebugGroup("Draw Compute Output")

                    accumulationTargetIdx = 1 - accumulationTargetIdx
                    renderEncoder.setRenderPipelineState(pipelineState)
                    renderEncoder.setFragmentTexture(accumulationTargets[accumulationTargetIdx], index: 0)
                    renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
                    
                    renderEncoder.endEncoding()
                    
                    renderEncoder.popDebugGroup()

                    if let drawable = view.currentDrawable {
                        commandBuffer.present(drawable)
                    }
                }
           }

            commandBuffer.commit()
        }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        /// Respond to drawable size or orientation changes here

        self.size = size
        let aspect = Float(size.width) / Float(size.height)
        projectionMatrix = matrix_perspective_right_hand(fovyRadians: radians_from_degrees(65), aspectRatio:aspect, nearZ: 0.1, farZ: 100.0)
        
        let textureDesc = MTLTextureDescriptor()
        textureDesc.pixelFormat = .rgba32Float
        textureDesc.textureType = .type2D
        textureDesc.width = Int(size.width)
        textureDesc.height = Int(size.height)
        textureDesc.usage = [.shaderWrite, .shaderRead]
        
        accumulationTargets.removeAll()
        for _ in 0..<2 {
            accumulationTargets.append(self.device.makeTexture(descriptor: textureDesc)!)
        }
    }
}

// Generic matrix math utility functions
func matrix4x4_rotation(radians: Float, axis: SIMD3<Float>) -> matrix_float4x4 {
    let unitAxis = normalize(axis)
    let ct = cosf(radians)
    let st = sinf(radians)
    let ci = 1 - ct
    let x = unitAxis.x, y = unitAxis.y, z = unitAxis.z
    return matrix_float4x4.init(columns:(vector_float4(    ct + x * x * ci, y * x * ci + z * st, z * x * ci - y * st, 0),
                                         vector_float4(x * y * ci - z * st,     ct + y * y * ci, z * y * ci + x * st, 0),
                                         vector_float4(x * z * ci + y * st, y * z * ci - x * st,     ct + z * z * ci, 0),
                                         vector_float4(                  0,                   0,                   0, 1)))
}

func matrix4x4_translation(_ translationX: Float, _ translationY: Float, _ translationZ: Float) -> matrix_float4x4 {
    return matrix_float4x4.init(columns:(vector_float4(1, 0, 0, 0),
                                         vector_float4(0, 1, 0, 0),
                                         vector_float4(0, 0, 1, 0),
                                         vector_float4(translationX, translationY, translationZ, 1)))
}

func matrix_perspective_right_hand(fovyRadians fovy: Float, aspectRatio: Float, nearZ: Float, farZ: Float) -> matrix_float4x4 {
    let ys = 1 / tanf(fovy * 0.5)
    let xs = ys / aspectRatio
    let zs = farZ / (nearZ - farZ)
    return matrix_float4x4.init(columns:(vector_float4(xs,  0, 0,   0),
                                         vector_float4( 0, ys, 0,   0),
                                         vector_float4( 0,  0, zs, -1),
                                         vector_float4( 0,  0, zs * nearZ, 0)))
}

func radians_from_degrees(_ degrees: Float) -> Float {
    return (degrees / 180) * .pi
}
