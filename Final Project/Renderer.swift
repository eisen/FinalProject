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

public enum ScalarSize {
    case BITS_8
    case BITS_16
}

class Renderer: NSObject, MTKViewDelegate {

    public let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var dynamicUniformBuffer: MTLBuffer
    var pipelineState: MTLRenderPipelineState
    var rtPipelineState: MTLComputePipelineState?
    var depthState: MTLDepthStencilState
    var colorMap: MTLTexture
    var octreeUInt8: Octree<Brick<UInt8>, UInt8>?
    var bricks8Cut: [Brick<UInt8>] = []
    var octreeUInt16: Octree<Brick<UInt16>, UInt16>?
    var bricks16Cut: [Brick<UInt16>] = []
    
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
    
    var accStruct: MTLAccelerationStructure?
    
    var scratchBuffer: MTLBuffer?
    var bboxBuffer: MTLBuffer?
    
    var size: CGSize?
    
    //var macroCellPool: MTLTexture?
    
    public func SetScalarSize(size: ScalarSize) {
        self.scalarSize = size
    }
    
    public func setOctreeUInt8(ocTree: Octree<Brick<UInt8>, UInt8>) {
        self.octreeUInt8 = ocTree
        self.scalarSize = .BITS_8
        
        let brickPoolDesc = MTLTextureDescriptor.init()
        
        brickPoolDesc.textureType = .type3D
        brickPoolDesc.height = 128 * 8
        brickPoolDesc.width = 128 * 8
        brickPoolDesc.depth = 128 * 9
        brickPoolDesc.pixelFormat = .r8Uint
        
        self.brickPool = self.device.makeTexture(descriptor: brickPoolDesc)!
        self.brickPool?.label = "Brick Pool UInt8"
        
        //        let macroCellPoolDesc = MTLTextureDescriptor.init()
        //
        //        macroCellPoolDesc.textureType = .type3D
        //        macroCellPoolDesc.height = 128 * 8
        //        macroCellPoolDesc.width = 128 * 8
        //        macroCellPoolDesc.depth = 128 * 9
        //        brickPoolDesc.pixelFormat = .r32Uint
        //
        //        self.macroCellPool = self.device.makeTexture(descriptor: macroCellPoolDesc)!
    }
    
    public func setOctreeUInt16(ocTree: Octree<Brick<UInt16>, UInt16>) {
        self.octreeUInt16 = ocTree
        self.scalarSize = .BITS_16
        
        let brickPoolDesc = MTLTextureDescriptor.init()
        
        brickPoolDesc.textureType = .type3D
        brickPoolDesc.height = 128 * 8
        brickPoolDesc.width = 128 * 8
        brickPoolDesc.depth = 128 * 9
        brickPoolDesc.pixelFormat = .r16Uint
        
        self.brickPool = self.device.makeTexture(descriptor: brickPoolDesc)!
        self.brickPool?.label = "Brick Pool UInt16"
    }
    
    public func setFPSlabel(label: NSTextField) {
        self.fpsLabel = label
    }
    
    public func createAccelerationDesc() {
        var accStructs: [MTLAccelerationStructure] = []
        self.accStructDesc = MTLPrimitiveAccelerationStructureDescriptor()
        let geomDesc = MTLAccelerationStructureBoundingBoxGeometryDescriptor()
        
        var bbox = MTLAxisAlignedBoundingBox()
        
        bbox.min = MTLPackedFloat3Make(0, 0, 0)
        bbox.max = MTLPackedFloat3Make(2.55, 2.55, 2.55)
        
        self.bboxBuffer = self.device.makeBuffer(length: MemoryLayout<MTLAxisAlignedBoundingBox>.size, options: .storageModeShared)
        memcpy(self.bboxBuffer?.contents(), &bbox, 1);
        
        geomDesc.boundingBoxBuffer = bboxBuffer
        geomDesc.boundingBoxCount = 1
        
        self.accStructDesc?.geometryDescriptors = [ geomDesc ]
        
        let sizes = self.device.accelerationStructureSizes(descriptor: accStructDesc!)
        self.accStruct = self.device.makeAccelerationStructure(size: sizes.accelerationStructureSize)
        self.accStruct?.label = "Acceleration Structure"
        self.scratchBuffer = self.device.makeBuffer(length: sizes.buildScratchBufferSize, options: .storageModePrivate)
        self.scratchBuffer?.label = "Scratch Buffer"
        
        accStructs.append(self.accStruct!)
        
        let instanceBuffer = device.makeBuffer(length: MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.size, options: .storageModeManaged)
        let instanceDesc: UnsafeMutablePointer<MTLAccelerationStructureInstanceDescriptor> = (instanceBuffer?.contents().assumingMemoryBound(to: MTLAccelerationStructureInstanceDescriptor.self))!
        
        instanceDesc[0].accelerationStructureIndex = 0
        instanceDesc[0].options = .nonOpaque
        instanceDesc[0].intersectionFunctionTableOffset = 0
        
        var destCols = instanceDesc[0].transformationMatrix.columns
        let srcCols = self.uniforms[0].modelViewMatrix.columns
        
        destCols.0 = MTLPackedFloat3Make(srcCols.0.x, srcCols.0.y, srcCols.0.z)
        destCols.1 = MTLPackedFloat3Make(srcCols.1.x, srcCols.1.y, srcCols.1.z)
        destCols.2 = MTLPackedFloat3Make(srcCols.2.x, srcCols.2.y, srcCols.2.z)
        destCols.3 = MTLPackedFloat3Make(srcCols.3.x, srcCols.3.y, srcCols.3.z)
        instanceDesc[0].transformationMatrix.columns = destCols

        let accelDescriptor = MTLInstanceAccelerationStructureDescriptor()
        
        accelDescriptor.instancedAccelerationStructures = accStructs
        accelDescriptor.instanceCount = 1
        accelDescriptor.instanceDescriptorBuffer = instanceBuffer
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

        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].format = MTLVertexFormat.float2
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].bufferIndex = BufferIndex.meshGenerics.rawValue

        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stride = 12
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepFunction = MTLVertexStepFunction.perVertex

        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stride = 8
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stepFunction = MTLVertexStepFunction.perVertex

        return mtlVertexDescriptor
    }

    class func buildRenderPipelineWithDevice(device: MTLDevice,
                                             metalKitView: MTKView,
                                             mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTLRenderPipelineState {
        /// Build a render state pipeline object

        let library = device.makeDefaultLibrary()

//        let vertexFunction = library?.makeFunction(name: "vertexShader")
//        let fragmentFunction = library?.makeFunction(name: "fragmentShader")
        
        let vertexFunction = library?.makeFunction(name: "copyVertex")
        let fragmentFunction = library?.makeFunction(name: "copyFragment")

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "RenderPipeline"
        //pipelineDescriptor.sampleCount = metalKitView.sampleCount
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        //pipelineDescriptor.vertexDescriptor = mtlVertexDescriptor

        pipelineDescriptor.colorAttachments[0].pixelFormat = metalKitView.colorPixelFormat
        pipelineDescriptor.depthAttachmentPixelFormat = metalKitView.depthStencilPixelFormat
        pipelineDescriptor.stencilAttachmentPixelFormat = metalKitView.depthStencilPixelFormat

        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }
    
    func buildComputePipelineWithDevice(device: MTLDevice,
                                             metalKitView: MTKView,
                                             mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTLComputePipelineState {
        let library = device.makeDefaultLibrary()!
        
        let constants = MTLFunctionConstantValues()
        var resourceStride = MemoryLayout<MTLAxisAlignedBoundingBox>.size
        var useIntersectionFunctions = true
        
        constants.setConstantValue(&resourceStride, type: .uint, index: 0)
        constants.setConstantValue(&useIntersectionFunctions, type: .bool, index: 1)
        
        let intersectionFunctionDesc = MTLIntersectionFunctionDescriptor()
        intersectionFunctionDesc.name = "brickIntersectionFunc"
        intersectionFunctionDesc.constantValues = constants
        let function = try library.makeIntersectionFunction(descriptor: intersectionFunctionDesc)
        
        let intersectionFunctionTableDesc = MTLIntersectionFunctionTableDescriptor()
        intersectionFunctionTableDesc.functionCount = 1

        let computePipelineDesc = MTLComputePipelineDescriptor()
        computePipelineDesc.computeFunction = library.makeFunction(name: "interceptBricks")!
        
        let linkedFunctions = MTLLinkedFunctions()
        linkedFunctions.functions = [function]
        computePipelineDesc.linkedFunctions = linkedFunctions
        
        let computePipelineState = try device.makeComputePipelineState(descriptor: computePipelineDesc, options: .bufferTypeInfo, reflection: nil)
        
        self.intersectionFunctionTable = computePipelineState.makeIntersectionFunctionTable(descriptor: intersectionFunctionTableDesc)
        self.intersectionFunctionTable?.setBuffer(self.bboxBuffer, offset: 0, index: 0)
        
        let functionHandle = computePipelineState.functionHandle(function: function)
        self.intersectionFunctionTable?.setFunction(functionHandle, index: 0)
        
        return computePipelineState
    }

    class func buildMesh(device: MTLDevice,
                         mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTKMesh {
        /// Create and condition mesh data to feed into a pipeline using the given vertex descriptor

        let metalAllocator = MTKMeshBufferAllocator(device: device)

        let mdlMesh = MDLMesh.newBox(withDimensions: SIMD3<Float>(1.28, 1.28, 1.28),
                                     segments: SIMD3<UInt32>(1, 1, 1),
                                     geometryType: MDLGeometryType.quads,
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
        let viewMatrix = matrix4x4_translation(0.0, 0.0, -2.0)
        uniforms[0].modelViewMatrix = simd_mul(viewMatrix, modelMatrix)
        
        uniforms[0].width = UInt32(self.size!.width)
        uniforms[0].height = UInt32(self.size!.height)
        uniforms[0].camera.position = vector_float3(0, 0, -2)
        uniforms[0].camera.forward = vector_float3(0, 0, 1)
        uniforms[0].camera.up = vector_float3(0, 1, 0)
        uniforms[0].camera.right = vector_float3(1, 0, 0)
        //rotation += 0.01
    }
    
    private func generateWorkingSet() {
        let c: Double = 1
        switch scalarSize {
        case .BITS_8:
            if self.octreeUInt8 != nil {
                self.bricks8Cut = (self.octreeUInt8?.findNodeCut(cameraPos: vector_double3(0, 0, -2), c: c))!
            }
        case .BITS_16:
            if self.octreeUInt16 != nil {
                self.bricks16Cut = (self.octreeUInt16?.findNodeCut(cameraPos: vector_double3(0, 0, -2), c: c))!
            }
        }
    }
    
    private func transferWorkingSet() {
        switch scalarSize {
        case .BITS_8:
            for (idx, brick) in bricks8Cut.enumerated() {
                let x = idx % (8 * 9)
                let y = (idx / 8) % 9
                let z = idx / 64
                let region = MTLRegionMake3D(x, y, z, 128, 128, 128)
                self.brickPool!.replace(region: region, mipmapLevel: 0, slice: 0, withBytes: brick.scalars, bytesPerRow: 128, bytesPerImage: 128 * 128)
            }

       case .BITS_16:
            print("TBD")
        }
    }
    
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
            
            self.generateWorkingSet()
            if(self.brickPool != nil) {
                self.transferWorkingSet()
            }
            
            let accStructEncoder = commandBuffer.makeAccelerationStructureCommandEncoder()!
            
            accStructEncoder.build(accelerationStructure: self.accStruct!, descriptor: self.accStructDesc!, scratchBuffer: self.scratchBuffer!, scratchBufferOffset: 0)
            
            accStructEncoder.endEncoding()
            
            let width = Int(self.size!.width)
            let height = Int(self.size!.height)
            
            let threadsPerThreadgroup = MTLSizeMake(8, 8, 1)
            let threadgroups = MTLSizeMake((width  + threadsPerThreadgroup.width  - 1) / threadsPerThreadgroup.width,
                                               (height + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
                                               1);
            
            let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
            
            computeEncoder.setAccelerationStructure(self.accStruct!, bufferIndex: 0)
            computeEncoder.setIntersectionFunctionTable(self.intersectionFunctionTable, bufferIndex: 1)
            
            computeEncoder.setBuffer(dynamicUniformBuffer, offset: 0, index: 2)
            
            computeEncoder.setComputePipelineState(rtPipelineState!)
            
            computeEncoder.setTexture(accumulationTargets[accumulationTargetIdx], index: 0)
            computeEncoder.setTexture(self.brickPool, index: TextureIndex.BrickPoolIndex.rawValue)
            computeEncoder.useResource(self.accStruct!, usage: .read)
            
            computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            
            computeEncoder.endEncoding()
            
            /// Delay getting the currentRenderPassDescriptor until we absolutely need it to avoid
            ///   holding onto the drawable and blocking the display pipeline any longer than necessary
            let renderPassDescriptor = view.currentRenderPassDescriptor

            if let renderPassDescriptor = renderPassDescriptor {

                /// Final pass rendering code here
                if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
                    renderEncoder.label = "Primary Render Encoder"

                    renderEncoder.pushDebugGroup("Draw Compute Output")
//
//                    renderEncoder.setCullMode(.front)
//
//                    renderEncoder.setFrontFacing(.counterClockwise)
//
//                    renderEncoder.setRenderPipelineState(pipelineState)
//
//                    renderEncoder.setDepthStencilState(depthState)
//
//                    renderEncoder.setVertexBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
//                    renderEncoder.setFragmentBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
//
//                    for (index, element) in mesh.vertexDescriptor.layouts.enumerated() {
//                        guard let layout = element as? MDLVertexBufferLayout else {
//                            return
//                        }
//
//                        if layout.stride != 0 {
//                            let buffer = mesh.vertexBuffers[index]
//                            renderEncoder.setVertexBuffer(buffer.buffer, offset:buffer.offset, index: index)
//                        }
//                    }
//
//                    renderEncoder.setFragmentTexture(colorMap, index: TextureIndex.TextureIndexColor.rawValue)
//                    renderEncoder.setFragmentTexture(self.brickPool, index: TextureIndex.BrickPoolIndex.rawValue)
//
//                    for submesh in mesh.submeshes {
//                        renderEncoder.drawIndexedPrimitives(type: submesh.primitiveType,
//                                                            indexCount: submesh.indexCount,
//                                                            indexType: submesh.indexType,
//                                                            indexBuffer: submesh.indexBuffer.buffer,
//                                                            indexBufferOffset: submesh.indexBuffer.offset)
//
//                    }
//
                    renderEncoder.popDebugGroup()

                    accumulationTargetIdx = 1 - accumulationTargetIdx
                    renderEncoder.setRenderPipelineState(pipelineState)
                    renderEncoder.setFragmentTexture(accumulationTargets[accumulationTargetIdx], index: 0)
                    renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
                    renderEncoder.endEncoding()

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
