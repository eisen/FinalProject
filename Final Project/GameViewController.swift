//
//  GameViewController.swift
//  Final Project
//
//  Created by Eisen Montalvo on 4/3/22.
//

import Cocoa
import MetalKit

// Our macOS specific view controller
class GameViewController: NSViewController, NSOpenSavePanelDelegate {
    
    var renderer: Renderer!
    var mtkView: MTKView!
    
    let rawView = RawView()
    
    @IBOutlet var progressBar: NSProgressIndicator!
    @IBOutlet var progressStatus: NSTextField!
    @IBOutlet var fpsLabel: NSTextField!
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        guard let mtkView = self.view as? MTKView else {
            print("View attached to GameViewController is not an MTKView")
            return
        }
        
        self.mtkView = mtkView
        
        // Select the device to render with.  We choose the default device
        guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return
        }
        
        mtkView.device = defaultDevice
        
        guard let newRenderer = Renderer(metalKitView: mtkView) else {
            print("Renderer cannot be initialized")
            return
        }
        
        renderer = newRenderer
        renderer.setFPSlabel(label: fpsLabel)
        
        renderer.mtkView(mtkView, drawableSizeWillChange: mtkView.drawableSize)
        
        mtkView.delegate = renderer
    }
    
    @IBAction
    func selectFile(_ sender: NSObject) {
        let myFileDialog = NSOpenPanel()
        
        myFileDialog.accessoryView = rawView
        myFileDialog.delegate = self
        myFileDialog.isAccessoryViewDisclosed = true
        myFileDialog.allowsMultipleSelection = false
        myFileDialog.allowedFileTypes = ["raw"]
        
        
        if myFileDialog.runModal() == .OK {
            let url = myFileDialog.url!
            let width = rawView.GetWidth()
            let height = rawView.GetHeight()
            let depth = rawView.GetDepth()
            
            ShowProgress()
            
            switch rawView.GetScalarSize() {
            case .BITS_16:
                self.renderer.SetScalarSize(size: .BITS_16)
                self.loadVolumeSet16(datFile: url, width: width, height: height, depth: depth)
            case .BITS_8:
                self.renderer.SetScalarSize(size: .BITS_8)
                self.loadVolumeSet8(datFile: url, width: width, height: height, depth: depth)
            }
        } else {
            HideProgress()
        }
    }
    
    private func ShowProgress() {
        progressBar.isHidden = false
        progressStatus.isHidden = false
    }
    
    private func HideProgress() {
        progressBar.isHidden = true
        progressStatus.isHidden = true
    }
    
    private func panelSelectionDidChange(_ sender: NSOpenPanel?) {
        let fileURL = sender!.url
        print("\(String(describing: fileURL))")
    }
    
    func loadVolumeSet16(datFile: URL, width: Int, height: Int, depth: Int) {
        var data: Data?
        do {
            data = try Data.init(contentsOf: datFile)
        } catch {
            print(error)
        }
        
        if ( width * height * depth * 2 != data?.count) {
            ShowAlert()
            return
        }
        
        let brickDim = 128
        
        var maxDim = max(width, height, depth)
        
        maxDim = Int(pow(2.0, ceil(log(Double(maxDim))/log(2))))
        
        let bricksPerDim = maxDim / brickDim
        
        let boxDim = Double(maxDim - 1)
        let bBox = Box(boxMin: vector3(0,0,0), boxMax: vector3(boxDim, boxDim, boxDim))
        let ocTree: Octree<Brick<UInt16>, UInt16> = Octree<Brick<UInt16>, UInt16>(boundingBox: bBox, minimumCellSize: Double(brickDim))
        
        let brickCenter = Double(brickDim / 2)
        var bricks: [Brick<UInt16>] = []
        
        progressBar.doubleValue = 0
        let bricksCount = pow(Double(bricksPerDim), Double(3))
        progressBar.maxValue = Double( bricksCount + 3)
        progressStatus.stringValue = "Extracting brick 1 of \(Int(bricksCount))..."
        
        DispatchQueue.global(qos: .userInitiated).async {

            let start = CFAbsoluteTimeGetCurrent()
            
            for xStep in 0..<bricksPerDim {
                for yStep in 0..<bricksPerDim {
                    for zStep in 0..<bricksPerDim {
                        let xCenter = Double(xStep * brickDim) + brickCenter
                        let yCenter = Double(yStep * brickDim) + brickCenter
                        let zCenter = Double(zStep * brickDim) + brickCenter
                        let brick = Brick<UInt16>(dataset: data, datasetSize: vector3(Double(width), Double(height), Double(depth)), position: vector3(xCenter, yCenter, zCenter), size: brickDim)
                        bricks.append(brick)
                        DispatchQueue.main.async {
                            self.progressBar.doubleValue += 1
                            self.progressStatus.stringValue = String(format:"Extracting brick %.0f of %.0f...", self.progressBar.doubleValue + 1, bricksCount)
                        }
                    }
                }
            }
            
            DispatchQueue.main.async {
                self.progressStatus.stringValue = "Adding bricks to Octree..."
            }
            for idx in 0..<bricks.count {
                ocTree.add(bricks[idx], at: bricks[idx].getPosition())
                
            }
            
            DispatchQueue.main.async {
                self.progressBar.doubleValue += 1
            }
            
            DispatchQueue.main.async {
                self.progressStatus.stringValue = "Generating brick LODs..."
            }
            
            ocTree.generateLODs()
            
            DispatchQueue.main.async {
                self.progressBar.doubleValue += 1
            }
            
            let end = CFAbsoluteTimeGetCurrent()
            print("Took \(end-start) seconds")
            
//            DispatchQueue.main.async {
//                self.progressStatus.stringValue = "Exporting bricks..."
//            }
//
//            ocTree.exportBricks(folder: datFile.baseURL!)
            
            DispatchQueue.main.async {
                self.HideProgress()
            }
            
            //print(ocTree)
            
            self.renderer.setOctreeUInt16(ocTree: ocTree)
        }
    }
    
    private func ShowAlert() {
        let alert = NSAlert()
        
        alert.alertStyle = .critical
        
        alert.messageText = "Error opening raw file!"
        alert.informativeText = "Invalid dimensions and/or scalar type for file. Aborting!"
        
        alert.runModal()
        HideProgress()
    }
    
    func loadVolumeSet8(datFile: URL, width: Int, height: Int, depth: Int) {
        
        progressStatus.stringValue = "Converting \(datFile.path)..."
        
        var data: Data?
        do{
            data = try Data.init(contentsOf: datFile)
        } catch {
            print(error)
        }
        
        if ( width * height * depth != data?.count) {
            ShowAlert()
            return
        }
        
        let brickDim = 128
        
        var maxDim = max(width, height, depth)
        
        maxDim = Int(pow(2.0, ceil(log(Double(maxDim))/log(2))))
        
        let bricksPerDim = maxDim / brickDim
        
        let boxDim = Double(maxDim - 1)
        let bBox = Box(boxMin: vector3(0,0,0), boxMax: vector3(boxDim, boxDim, boxDim))
        let ocTree: Octree<Brick<UInt8>, UInt8> = Octree<Brick<UInt8>, UInt8>(boundingBox: bBox, minimumCellSize: Double(brickDim))
        
        let brickCenter = Double(brickDim / 2)
        var bricks: [Brick<UInt8>] = []
        
        progressBar.doubleValue = 0
        let bricksCount = pow(Double(bricksPerDim), Double(3))
        progressBar.maxValue = Double( bricksCount + 3)
        progressStatus.stringValue = "Extracting brick 1 of \(Int(bricksCount))..."
        
        DispatchQueue.global(qos: .userInitiated).async {
            let start = CFAbsoluteTimeGetCurrent()
            
            for xStep in 0..<bricksPerDim {
                for yStep in 0..<bricksPerDim {
                    for zStep in 0..<bricksPerDim {
                        let xCenter = Double(xStep * brickDim) + brickCenter
                        let yCenter = Double(yStep * brickDim) + brickCenter
                        let zCenter = Double(zStep * brickDim) + brickCenter
                        let brick = Brick<UInt8>(dataset: data, datasetSize: vector3(Double(width), Double(height), Double(depth)), position: vector3(xCenter, yCenter, zCenter), size: brickDim)
                        bricks.append(brick)
                        DispatchQueue.main.async {
                            self.progressBar.doubleValue += 1
                            self.progressStatus.stringValue = String(format:"Extracting brick %.0f of %.0f...", self.progressBar.doubleValue, bricksCount)
                        }
                    }
                }
            }
            
            DispatchQueue.main.async {
                self.progressStatus.stringValue = "Adding bricks to Octree..."
            }
            for idx in 0..<bricks.count {
                ocTree.add(bricks[idx], at: bricks[idx].getPosition())
            }
            DispatchQueue.main.async {
                self.progressBar.doubleValue += 1
            }
            
            DispatchQueue.main.async {
                self.progressStatus.stringValue = "Generating brick LODs..."
            }
            ocTree.generateLODs()
            DispatchQueue.main.async {
                self.progressBar.doubleValue += 1
            }
            
            let end = CFAbsoluteTimeGetCurrent()
            print("Took \(end-start) seconds")
            
//            DispatchQueue.main.async {
//                self.progressStatus.stringValue = "Exporting bricks..."
//            }
//
//            ocTree.exportBricks(folder: URL(fileURLWithPath: datFile.path).deletingLastPathComponent())
            
            DispatchQueue.main.async {
                self.HideProgress()
            }
            
            //print(ocTree)
            
            self.renderer.setOctreeUInt8(ocTree: ocTree)
        }
    }
}
