//
//  Brick.swift
//  Final Project
//
//  Created by Nishita Kharche & Eisen Montalvo on 4/8/22.
//

import Foundation
import simd

public class Brick<T: FixedWidthInteger>: CustomStringConvertible, Equatable {
    public static func == (lhs: Brick, rhs: Brick) -> Bool {
        return lhs.getPosition() == rhs.getPosition()
    }
    
    var min_scalar: T
    var max_scalar: T
    let dataset: Data?
    let datasetSize: vector_double3?
    var position: vector_double3 = vector_double3(0,0,0)
    var size: Int = 128
    var halfSize: Int = 64
    var scalars: [T]! = []
    var macroCells: [MacroCell<T>] = []
    
    public init(dataset: Data?, datasetSize: vector_double3, position: vector_double3, size: Int) {
        self.dataset = dataset
        self.datasetSize = datasetSize
        self.position = position
        self.size = size
        self.halfSize = size >> 1
        self.min_scalar = T.max
        self.max_scalar = T.min
        
        if(dataset != nil) {
            //DispatchQueue.global(qos: .userInitiated).async {
                self.extract()
                if self.scalars.count > 0 {
                    self.generateMacroCells()
                }
            //}
        }
    }
    
    public func createBrick(bricks: [Brick<T>]) {
        self.generateLOD(bricks: bricks)
        self.generateMacroCells()
    }
    
    func initMacroCells() {
        let mcDimSize = self.size >> 3
        let mcCount = mcDimSize * mcDimSize * mcDimSize
        for _ in 0..<mcCount {
            self.macroCells.append(MacroCell<T>())
        }
    }
    
    public var description: String {
        let scalarFormatted = ByteCountFormatter()
        let scalarCount = scalarFormatted.string(fromByteCount: Int64(scalars.count))
        return "Brick @ (\(position)) [\(scalarCount)] \(self.min_scalar)...\(self.max_scalar) MC \(self.macroCells.count)"
    }
    
    func extract() {
        let halfSize: vector_double3 = vector_double3(Double(self.halfSize), Double(self.halfSize), Double(self.halfSize))
        let start = self.position - halfSize
        let end = self.position + halfSize
        let typeStride = Int(T.bitWidth >> 3) // Divide by 8 to find bytes
        
        for dz in Int(start.z) ..< Int(end.z) {
            let zOffset = dz * Int(self.datasetSize!.x) * Int(self.datasetSize!.y)
            if(dz < Int(self.datasetSize!.z) ) {
                for dy in Int(start.y) ..< Int(end.y) {
                    if dy < Int(self.datasetSize!.y) {
                        let yOffset = dy * Int(self.datasetSize!.x)
                        let range: Range<Data.Index> = Range<Data.Index>(uncheckedBounds: (lower: (zOffset + yOffset + Int(start.x)) * typeStride, upper: (zOffset + yOffset + Int(end.x)) * typeStride))
                        if range.upperBound < self.dataset?.count ?? 0 {
                            let subdata = self.dataset!.subdata(in: range)
                             
                            for idx in stride(from: 0, to: subdata.count, by: typeStride) {
                                let dIdx: Int = subdata.index(idx, offsetBy: 0)
                                let scalarRange: Range<Data.Index> = Range<Data.Index>(uncheckedBounds: (lower: dIdx, upper: dIdx + typeStride))
                                var scalar: T = T.min
                                _ = withUnsafeMutableBytes(of: &scalar) { bufPtr in
                                    subdata.copyBytes(to: bufPtr, from: scalarRange).littleEndian
                                }
                                scalars.append(scalar)
                            }
                        } else {
                            for _ in Int(start.x) ..< Int(end.x) {
                                scalars.append(T.min)
                            }
                        }
                    } else {
                        for _ in Int(start.x) ..< Int(end.x) {
                            scalars.append(T.min)
                        }
                    }
                }
            } else {
                for _ in Int(start.y) ..< Int(end.y) {
                    for _ in Int(start.x) ..< Int(end.x) {
                        scalars.append(T.min)
                    }
                }
            }
        }
        
        if(scalars.count > 0) {
            self.min_scalar = scalars.min()!
            self.max_scalar = scalars.max()!
        } else {
            min_scalar = T.min
            max_scalar = T.min
        }
    }
    
    func generateLOD(bricks: [Brick<T>]) {
        for dz in 0..<self.size {
            let offsetZ = (dz & 0x40) >> 4
            for dy in 0..<self.size {
                let offsetY = (dy & 0x40) >> 5
                for dx in 0..<self.size {
                    let offsetX = (dx & 0x40) >> 6
                    let brickIdx = Int(offsetX + offsetY + offsetZ)
                    let brick = bricks[brickIdx]
                    let scalar = brick.getBlockAverage(position: vector_double3(Double(dx & 0x3F), Double(dy & 0x3F), Double(dz & 0x3F)))
                    self.scalars.append(scalar)
                }
            }
        }
    }
    
    func getBlockAverage(position: vector_double3) -> T {
        let start: vector_double3 = 2 * position
        let end = start + 1
        var result: T = 0
        
        for dz in Int(start.z) ... Int(end.z) {
            let zOffset = dz * Int(self.size) * Int(self.size)
            for dy in Int(start.y) ... Int(end.y) {
                let yOffset = dy * Int(self.size)
                for dx in Int(start.x) ... Int(end.x) {
                    let scalarIdx = zOffset + yOffset + dx
                    if(scalarIdx < self.scalars.count) {
                        let scalar = self.scalars[scalarIdx]
                        result += scalar >> 3 // Divide by 8 to find average of cells and avoid overflow of type
                    }
                }
            }
        }
        
        return result
    }
    
    func generateMacroCells() {
        initMacroCells()
        for dz in 0..<self.size {
            let zOffset = dz * self.size * self.size
            let mcZoffset = (dz >> 3) * 16 * 16
            for dy in 0..<self.size {
                let yOffset = dy * self.size
                let mcYoffset = (dy >> 3) * 16
                for dx in 0..<self.size {
                    let mcIdx = Int(mcZoffset + mcYoffset + (dx >> 3))
                    self.macroCells[mcIdx].addScalar(scalar: scalars[Int(zOffset + yOffset + dx)])
                }
            }
        }
    }
    
    public func exportRaw(fileURL: URL) {
        do {
            let brickData = Data(bytes: self.scalars, count: scalars.count * MemoryLayout<T>.stride)
            try brickData.write(to: fileURL)
        } catch {
            print(error)
        }
    }
    
    public func getPosition() -> vector_double3 {
        return self.position
    }
    
    public func isEmpty() -> Bool {
        return min_scalar == 0 && max_scalar == 0
    }
    
    public func printMacrocells() {
        for idx in 0..<self.macroCells.count {
            print("\(idx): \(self.macroCells[idx])")
        }
    }
}
