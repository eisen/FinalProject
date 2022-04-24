//
//  Octree.swift
//  Final Project
//
//  Created by Nishita Kharche & Eisen Montalvo on 4/8/22.
//

import Foundation
import simd

public struct Box: CustomStringConvertible {
    public var boxMin: vector_double3
    public var boxMax: vector_double3
    
    public init(boxMin: vector_double3, boxMax: vector_double3) {
        self.boxMin = boxMin
        self.boxMax = boxMax
    }
    
    public var boxSize: vector_double3 {
        return boxMax - boxMin
    }
    
    var halfBoxSize: vector_double3 {
        return boxSize/2
    }
    
    var frontLeftTop: Box {
        let boxMin = self.boxMin + vector_double3(0, halfBoxSize.y, halfBoxSize.z)
        let boxMax = self.boxMax - vector_double3(halfBoxSize.x, 0, 0)
        return Box(boxMin: boxMin, boxMax: boxMax)
    }
    var frontLeftBottom: Box {
        let boxMin = self.boxMin + vector_double3(0, 0, halfBoxSize.z)
        let boxMax = self.boxMax - vector_double3(halfBoxSize.x, halfBoxSize.y, 0)
        return Box(boxMin: boxMin, boxMax: boxMax)
    }
    var frontRightTop: Box {
        let boxMin = self.boxMin + vector_double3(halfBoxSize.x, halfBoxSize.y, halfBoxSize.z)
        let boxMax = self.boxMax - vector_double3(0, 0, 0)
        return Box(boxMin: boxMin, boxMax: boxMax)
    }
    var frontRightBottom: Box {
        let boxMin = self.boxMin + vector_double3(halfBoxSize.x, 0, halfBoxSize.z)
        let boxMax = self.boxMax - vector_double3(0, halfBoxSize.y, 0)
        return Box(boxMin: boxMin, boxMax: boxMax)
    }
    var backLeftTop: Box {
        let boxMin = self.boxMin + vector_double3(0, halfBoxSize.y, 0)
        let boxMax = self.boxMax - vector_double3(halfBoxSize.x, 0, halfBoxSize.z)
        return Box(boxMin: boxMin, boxMax: boxMax)
    }
    var backLeftBottom: Box {
        let boxMin = self.boxMin + vector_double3(0, 0, 0)
        let boxMax = self.boxMax - vector_double3(halfBoxSize.x, halfBoxSize.y, halfBoxSize.z)
        return Box(boxMin: boxMin, boxMax: boxMax)
    }
    var backRightTop: Box {
        let boxMin = self.boxMin + vector_double3(halfBoxSize.x, halfBoxSize.y, 0)
        let boxMax = self.boxMax - vector_double3(0, 0, halfBoxSize.z)
        return Box(boxMin: boxMin, boxMax: boxMax)
    }
    var backRightBottom: Box {
        let boxMin = self.boxMin + vector_double3(halfBoxSize.x, 0, 0)
        let boxMax = self.boxMax - vector_double3(0, halfBoxSize.y, halfBoxSize.z)
        return Box(boxMin: boxMin, boxMax: boxMax)
    }
    
    public func contains(_ point: vector_double3) -> Bool {
        return (boxMin.x <= point.x && point.x <= boxMax.x) && (boxMin.y <= point.y && point.y <= boxMax.y) && (boxMin.z <= point.z && point.z <= boxMax.z)
    }
    
    public func contains(_ box: Box) -> Bool {
        return
            self.boxMin.x <= box.boxMin.x &&
                self.boxMin.y <= box.boxMin.y &&
                self.boxMin.z <= box.boxMin.z &&
                self.boxMax.x >= box.boxMax.x &&
                self.boxMax.y >= box.boxMax.y &&
                self.boxMax.z >= box.boxMax.z
    }
    
    public func isContained(in box: Box) -> Bool {
        return
            self.boxMin.x >= box.boxMin.x &&
                self.boxMin.y >= box.boxMin.y &&
                self.boxMin.z >= box.boxMin.z &&
                self.boxMax.x <= box.boxMax.x &&
                self.boxMax.y <= box.boxMax.y &&
                self.boxMax.z <= box.boxMax.z
    }
    
    /* This intersect function does not handle all possibilities such as two beams
     of different diameter crossing each other half way. But it does cover all cases
     needed for an octree as the bounding box has to contain the given intersect box */
    public func intersects(_ box: Box) -> Bool {
        let corners = [
            vector_double3(boxMin.x, boxMax.y, boxMax.z), //frontLeftTop
            vector_double3(boxMin.x, boxMin.y, boxMax.z), //frontLeftBottom
            vector_double3(boxMax.x, boxMax.y, boxMax.z), //frontRightTop
            vector_double3(boxMax.x, boxMin.y, boxMax.z), //frontRightBottom
            vector_double3(boxMin.x, boxMax.y, boxMin.z), //backLeftTop
            vector_double3(boxMin.x, boxMin.y, boxMin.z), //backLeftBottom
            vector_double3(boxMax.x, boxMax.y, boxMin.z), //backRightTop
            vector_double3(boxMax.x, boxMin.y, boxMin.z)  //backRightBottom
        ]
        for corner in corners {
            if box.contains(corner) {
                return true
            }
        }
        return false
    }
    
    public var description: String {
        return "Box from:\(boxMin) to:\(boxMax)"
    }
}

public class OctreeNode<T: Equatable, V: FixedWidthInteger>: CustomStringConvertible {
    let box: Box
    var point: vector_double3!
    var elements: [T]! = []
    var type: NodeType = .leaf
    let docDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    
    let sortOrder: [[Int]] = [
        [ 0,    2,    1,    4,    3,    6,    5,    7 ],
        [ 1,    3,    0,    5,    2,    7,    4,    6 ],
        [ 2,    3,    0,    6,    1,    7,    4,    5 ],
        [ 3,    2,    1,    7,    0,    6,    5,    4 ],
        [ 4,    0,    6,    5,    2,    1,    7,    3 ],
        [ 5,    1,    7,    4,    3,    0,    6,    2 ],
        [ 6,    3,    0,    5,    2,    7,    4,    1 ],
        [ 7,    3,    6,    5,    2,    1,    4,    0 ]
    ]
    
    enum NodeType {
        case leaf
        case `internal`(children: Children)
    }
    
    public var description: String {
        switch type {
            case .leaf:
                return "leaf node with \(box) elements: \(String(describing: elements))"
            case .internal:
                return "internal node with \(box) elements: \(String(describing: elements))"
        }
    }
    
    var recursiveDescription: String {
        return recursiveDescription(withTabCount: 0)
    }
    
    private func recursiveDescription(withTabCount count: Int) -> String {
        let indent = String(repeating: "\t", count: count)
        var result = "\(indent)" + description + "\n"
        switch type {
        case .internal(let children):
            for child in children {
                result += child.recursiveDescription(withTabCount: count + 1)
            }
        default:
            break
        }
        return result
    }
    
    struct Children: Sequence {
        let frontLeftTop: OctreeNode
        let frontLeftBottom: OctreeNode
        let frontRightTop: OctreeNode
        let frontRightBottom: OctreeNode
        let backLeftTop: OctreeNode
        let backLeftBottom: OctreeNode
        let backRightTop: OctreeNode
        let backRightBottom: OctreeNode
        
        init(parentNode: OctreeNode) {
            frontLeftTop = OctreeNode(box: parentNode.box.frontLeftTop)
            frontLeftBottom = OctreeNode(box: parentNode.box.frontLeftBottom)
            frontRightTop = OctreeNode(box: parentNode.box.frontRightTop)
            frontRightBottom = OctreeNode(box: parentNode.box.frontRightBottom)
            backLeftTop = OctreeNode(box: parentNode.box.backLeftTop)
            backLeftBottom = OctreeNode(box: parentNode.box.backLeftBottom)
            backRightTop = OctreeNode(box: parentNode.box.backRightTop)
            backRightBottom = OctreeNode(box: parentNode.box.backRightBottom)
        }
        
        struct ChildrenIterator: IteratorProtocol {
            var index = 0
            let children: Children
            
            init(children: Children) {
                self.children = children
            }
            
            mutating func next() -> OctreeNode? {
                defer { index += 1 }
                switch index {
                    case 0: return children.backLeftBottom
                    case 1: return children.backRightBottom
                    case 2: return children.backLeftTop
                    case 3: return children.backRightTop
                    case 4: return children.frontLeftBottom
                    case 5: return children.frontRightBottom
                    case 6: return children.frontLeftTop
                    case 7: return children.frontRightTop
                    
                    default: return nil
                }
            }
        }
        
        func makeIterator() -> ChildrenIterator {
            return ChildrenIterator(children: self)
        }
        
//        func getChildByIndex(index: Int) {
//            switch index {
//                case 0: return children.backLeftBottom
//                case 1: return children.backRightBottom
//                case 2: return children.backLeftTop
//                case 3: return children.backRightTop
//                case 4: return children.frontLeftBottom
//                case 5: return children.frontRightBottom
//                case 6: return children.frontLeftTop
//                case 7: return children.frontRightTop
//
//                default: return nil
//            }
//        }
    }
    
    init(box: Box) {
        self.box = box
    }
    
    @discardableResult
    func add(_ element: T, at point: vector_double3) -> OctreeNode {
        return tryAdd(element, at: point)!
    }
    
    private func tryAdd(_ element: T, at point: vector_double3) -> OctreeNode? {
        if !box.contains(point) {
            return nil
        }
        
        switch type {
        case .internal(let children):
            // pass the point to one of the children
            for child in children {
                if let child = child.tryAdd(element, at: point) {
                    return child
                }
            }
            
            fatalError("box.contains evaluted to true, but none of the children added the point")
        case .leaf:
            if self.point != nil {
                // leaf already has an asigned point
                if self.point == point {
                    self.elements.append(element)
                    return self
                } else {
                    return subdivide(adding: element, at: point)
                }
            } else {
                self.elements = [element]
                self.point = point
                return self
            }
        }
    }
    
    func add(_ elements: [T], at point: vector_double3) {
        for element in elements {
            self.add(element, at: point)
        }
    }
    
    @discardableResult
    func remove(_ element: T) -> Bool {
        switch type {
        case .leaf:
            if let elements = self.elements {
                // leaf contains one ore more elements
                if let index = elements.firstIndex(of: element) {
                    // leaf contains the element we want to remove
                    self.elements.remove(at: index)
                    // if elements is now empty remove it
                    if self.elements.isEmpty {
                        self.elements = nil
                    }
                    return true
                }
            }
            return false
        case .internal(let children):
            for child in children  {
                if child.remove(element) {
                    return true
                }
            }
            return false
        }
    }
    
    func generateBrick(nodes: Children) {
        var bricks: [T] = []
        for node in nodes {
            let brick: T = node.elements[0] // Only one Brick on leaf nodes
            bricks.append(brick)
        }
        let brick: Brick<V> = Brick<V>(dataset: nil, datasetSize: vector_double3(0,0,0), position: self.box.boxMin + self.box.halfBoxSize, size: 128)
        brick.createBrick(bricks: bricks as! [Brick<V>])
        if self.elements == nil {
            self.elements = []
        }
        if self.point == nil {
            self.point = self.box.halfBoxSize
        }
        self.elements.append(brick as! T)
    }
    
    func generateLODs() {
        if case .leaf = type {
            return
        }
        
        if case .internal(let children) = type {
            if areChildrenLeafs() == false {
                for child in children {
                    child.generateLODs()
                }
            }
            self.generateBrick(nodes:children)
        }
    }
    
    func areChildrenLeafs() -> Bool {
        switch type {
            case .leaf:
                return false // Leaf nodes don't need to calculate LODs
                
            case .internal(let children):
                for child in children {
                    if case .internal = child.type {
                        return false
                    }
                }
        }
        
        return true
    }
    
    func exportBrick(folder: URL, level: Int) {
        let brick = self.elements[0] as! Brick<V>
        let fileURL = folder.appendingPathComponent("brick_\(level)_\(brick.position.x)_\(brick.position.y)_\(brick.position.z).raw")
        print("Exporting \(fileURL)")
        brick.exportRaw(fileURL: fileURL)
    }
    
    func exportBricks(folder: URL, level: Int = 0) {
        switch type {
            case .leaf:
            self.exportBrick(folder: folder, level: level)
                
            case .internal(let children):
                for child in children {
                    child.exportBricks(folder: folder, level: level + 1)
                }
            self.exportBrick(folder: folder, level: level)
        }
    }
    
    func getBrick() -> T {
        return self.elements[0]
    }
    
    func elements(at point: vector_double3) -> [T]? {
        switch type {
            case .leaf:
                if self.point == point {
                    return self.elements
                }
            case .internal(let children):
                for child in children {
                    if child.box.contains(point) {
                        return child.elements(at: point)
                    }
                }
        }
        // tree does not contain given point
        return nil
    }
    
    func elements(in box: Box) -> [T]? {
        var values: [T] = []
        switch type {
        case .leaf:
            // check if leaf has an assigned point
            if let point = self.point {
                // check if point is inside given box
                if box.contains(point) {
                    values += elements ?? []
                }
            }
        case .internal(let children):
            for child in children {
                if child.box.isContained(in: box) {
                    // child is contained in box
                    // add all children of child
                    values += child.elements(in: child.box) ?? []
                } else if child.box.contains(box) || child.box.intersects(box) {
                    // child contains at least part of box
                    values += child.elements(in: box) ?? []
                }
                // child does not contain any part of given box
            }
        }
        if values.isEmpty { return nil }
        return values
    }
    
    private func subdivide(adding element: T, at point: vector_double3) -> OctreeNode? {
        precondition(self.elements != nil, "Subdividing while leaf does not contain a element")
        precondition(self.point != nil, "Subdividing while leaf does not contain a point")
        switch type {
        case .leaf:
            type = .internal(children: Children(parentNode: self))
            // add element previously contained in leaf to children
            self.add(self.elements, at: self.point)
            self.elements = nil
            self.point = nil
            // add new element to children
            return self.add(element, at: point)
        case .internal:
            preconditionFailure("Calling subdivide on an internal node")
        }
    }
    
    private func calculateImageSize(cameraPos: vector_double3, c: Double) -> Double {
        let originToCamera: vector_double3 = self.point - cameraPos
        let alpha = asin( 128 / distance(self.point, cameraPos))
        let beta = acos(dot(vector_double3(0, 0, 1), normalize(originToCamera)))
        
        let SC: Double = sin(alpha) * cos(beta)
        let CS: Double = cos(alpha) * sin(beta)
        let SS: Double = sin(alpha) * sin(beta)
        let CC: Double = sin(alpha) * cos(beta)
        
        let tanApB = (SC + CS) / (CC - SS)
        let tanAmB = (SC - CS) / (CC + SS)
        
        return c * abs(tanApB + tanAmB)
    }
    
    public func findNodeCut(cameraPos: vector_double3, c: Double) -> [T]! {
        var nodes: [T] = []
        
        switch self.type{
        case .leaf:
            return [self.elements[0]]
        case .internal(let children):
            var imageSize = self.calculateImageSize(cameraPos: cameraPos, c: c)
            
            if(imageSize < 128) {
                nodes.append(self.elements[0])
            } else {
                // Subdivide into higher resolution bricks
                for child in children {
                    imageSize = child.calculateImageSize(cameraPos: cameraPos, c: c)
                    
                    if(imageSize < 128) {
                        nodes.append(child.getBrick())
                    } else {
                        for node in child.findNodeCut(cameraPos: cameraPos, c: c) {
                            nodes.append(node)
                        }
                    }
                }
            }
        }
        
        return nodes
    }
}

public class Octree<T: Equatable, V: FixedWidthInteger>: CustomStringConvertible {
    var root: OctreeNode<T, V>
    
    public var description: String {
        return "Octree\n" + root.recursiveDescription
    }
    
    public init(boundingBox: Box, minimumCellSize: Double) {
        root = OctreeNode<T, V>(box: boundingBox)
    }
    
    @discardableResult
    public func add(_ element: T, at point: vector_double3) -> OctreeNode<T, V> {
        return root.add(element, at: point)
    }
    
    @discardableResult
    public func remove(_ element: T, using node: OctreeNode<T, V>) -> Bool {
        return node.remove(element)
    }
    
    @discardableResult
    public func remove(_ element: T) -> Bool {
        return root.remove(element)
    }
    
    public func generateLODs() {
        root.generateLODs()
    }
    
    public func exportBricks(folder: URL) {
        root.exportBricks(folder: folder)
    }
    
    public func elements(at point: vector_double3) -> [T]? {
        return root.elements(at: point)
    }
    
    public func elements(in box: Box) -> [T]? {
        precondition(root.box.contains(box), "box is outside of octree bounds")
        return root.elements(in: box)
    }
    
    public func findNodeCut(cameraPos: vector_double3, c: Double) -> [T]? {
        return root.findNodeCut(cameraPos: cameraPos, c: c)
    }
}
