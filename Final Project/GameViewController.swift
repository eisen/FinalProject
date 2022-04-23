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
    
    let defaults = UserDefaults.standard
    
    @IBOutlet var progressBar: NSProgressIndicator!
    @IBOutlet var progressStatus: NSTextField!
    @IBOutlet var fpsLabel: NSTextField!
    @IBOutlet var isoLabel: NSTextField!
    @IBOutlet var isoSlider: NSSlider!
    @IBOutlet var pitchSlider: NSSlider!
    @IBOutlet var yawSlider: NSSlider!

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
    func updateIsoValue(_ sender: NSSlider) {
        self.isoLabel.intValue = sender.intValue
        self.renderer.isoValue = Float(sender.intValue)
    }
    
    @IBAction
    func updatePitchValue(_ sender: NSSlider) {
        let value = (sender.floatValue / 50.0) * Float.pi
        self.renderer.rotation[0] = value
    }
    
    @IBAction
    func updateYawValue(_ sender: NSSlider) {
        let value = (sender.floatValue / 50.0) * Float.pi
        self.renderer.rotation[1] = value
    }
    
    @IBAction
    func selectFile(_ sender: NSObject) {

        let myFileDialog = NSOpenPanel()
        
        self.rawView.width.intValue = Int32(defaults.integer(forKey: "Width"))
        self.rawView.height.intValue = Int32(defaults.integer(forKey: "Height"))
        self.rawView.depth.intValue = Int32(defaults.integer(forKey: "Depth"))
        switch ScalarSize(rawValue: defaults.integer(forKey: "Scalar Size")) {
        case .BITS_8:
            self.rawView.byte.state = .on
        case .BITS_16:
            self.rawView.word.state = .on
        case .none:
            self.rawView.byte.state = .on
        }
        
        myFileDialog.accessoryView = self.rawView
        myFileDialog.delegate = self
        myFileDialog.isAccessoryViewDisclosed = true
        myFileDialog.allowsMultipleSelection = false
        myFileDialog.allowedFileTypes = ["raw"]
        
        if myFileDialog.runModal() == .OK {
            let url = myFileDialog.url!
            let width = Int32(self.rawView.GetWidth())
            let height = Int32(self.rawView.GetHeight())
            let depth = Int32(self.rawView.GetDepth())
            let scalarSize = self.rawView.GetScalarSize()
            
            defaults.set(width, forKey: "Width")
            defaults.set(height, forKey: "Height")
            defaults.set(depth, forKey: "Depth")
            defaults.set(scalarSize.rawValue, forKey: "Scalar Size")
            
            self.ShowProgress()
            
            //IsoFeatures.GetIsoValues(url: url.path, dim: vector_int3(x: Int32(width), y: Int32(height), z: Int32(depth)), scalarSize: self.rawView.GetScalarSize())
            
            switch scalarSize {
            case .BITS_16:
                //self.renderer.SetScalarSize(size: .BITS_16)
                self.isoSlider.maxValue = 4095
                self.isoSlider.intValue = 0
                self.isoLabel.intValue = 0
                self.renderer.isoValue = 0
                self.loadVolumeSet16(datFile: url, width: width, height: height, depth: depth)
            case .BITS_8:
                //self.renderer.SetScalarSize(size: .BITS_8)
                self.isoSlider.maxValue = 255
                self.isoSlider.intValue = 0
                self.isoLabel.intValue = 0
                self.renderer.isoValue = 0
                self.loadVolumeSet8(datFile: url, width: width, height: height, depth: depth)
            }
        } else {
            self.HideProgress()
        }
    }
    
    public func ShowProgress() {
        progressBar.isHidden = false
        progressStatus.isHidden = false
        progressBar.startAnimation(nil)
    }
    
    public func HideProgress() {
        progressBar.stopAnimation(nil)
        progressBar.isHidden = true
        progressStatus.isHidden = true
    }
    
    private func panelSelectionDidChange(_ sender: NSOpenPanel?) {
        let fileURL = sender!.url
        print("\(String(describing: fileURL))")
    }
    
    func loadVolumeSet16(datFile: URL, width: Int32, height: Int32, depth: Int32) {
        progressStatus.stringValue = "Loading \(datFile.lastPathComponent)..."
        
        var data: Data?
        do {
            data = try Data.init(contentsOf: datFile)
        } catch {
            print(error)
        }
        
        if ( width * height * depth * 2 != data!.count) {
            ShowAlert()
            return
        }
        
        self.renderer.setVolumeData16(data: data!, dim: vector_int3(width, height, depth))
//        
//        let brickDim = 128
//        
//        var maxDim = max(width, height, depth)
//        
//        maxDim = Int(pow(2.0, ceil(log(Double(maxDim))/log(2))))
//        
//        let bricksPerDim = maxDim / brickDim
//        
//        let boxDim = Double(maxDim - 1)
//        let bBox = Box(boxMin: vector3(0,0,0), boxMax: vector3(boxDim, boxDim, boxDim))
//        let ocTree: Octree<Brick<UInt16>, UInt16> = Octree<Brick<UInt16>, UInt16>(boundingBox: bBox, minimumCellSize: Double(brickDim))
//        
//        let brickCenter = Double(brickDim / 2)
//        var bricks: [Brick<UInt16>] = []
//        
//        progressBar.doubleValue = 0
//        let bricksCount = pow(Double(bricksPerDim), Double(3))
//        progressBar.maxValue = Double( bricksCount + 3)
//        progressStatus.stringValue = "Extracting brick 1 of \(Int(bricksCount))..."
//        
//        DispatchQueue.global(qos: .userInitiated).async {
//
//            let start = CFAbsoluteTimeGetCurrent()
//            
//            for xStep in 0..<bricksPerDim {
//                for yStep in 0..<bricksPerDim {
//                    for zStep in 0..<bricksPerDim {
//                        let xCenter = Double(xStep * brickDim) + brickCenter
//                        let yCenter = Double(yStep * brickDim) + brickCenter
//                        let zCenter = Double(zStep * brickDim) + brickCenter
//                        let brick = Brick<UInt16>(dataset: data, datasetSize: vector3(Double(width), Double(height), Double(depth)), position: vector3(xCenter, yCenter, zCenter), size: brickDim)
//                        bricks.append(brick)
//                        DispatchQueue.main.async {
//                            self.progressBar.doubleValue += 1
//                            self.progressStatus.stringValue = String(format:"Extracting brick %.0f of %.0f...", self.progressBar.doubleValue + 1, bricksCount)
//                        }
//                    }
//                }
//            }
//            
//            DispatchQueue.main.async {
//                self.progressStatus.stringValue = "Adding bricks to Octree..."
//            }
//            for idx in 0..<bricks.count {
//                ocTree.add(bricks[idx], at: bricks[idx].getPosition())
//                
//            }
//            
//            DispatchQueue.main.async {
//                self.progressBar.doubleValue += 1
//            }
//            
//            DispatchQueue.main.async {
//                self.progressStatus.stringValue = "Generating brick LODs..."
//            }
//            
//            ocTree.generateLODs()
//            
//            DispatchQueue.main.async {
//                self.progressBar.doubleValue += 1
//            }
//            
//            let end = CFAbsoluteTimeGetCurrent()
//            print("Took \(end-start) seconds")
//            
////            DispatchQueue.main.async {
////                self.progressStatus.stringValue = "Exporting bricks..."
////            }
////
////            ocTree.exportBricks(folder: datFile.baseURL!)
//            
//            DispatchQueue.main.async {
//                self.HideProgress()
//            }
//            
//            //print(ocTree)
//            
//            self.renderer.setOctreeUInt16(ocTree: ocTree)
//        }
    }
    
    private func ShowAlert() {
        let alert = NSAlert()
        
        alert.alertStyle = .critical
        
        alert.messageText = "Error opening raw file!"
        alert.informativeText = "Invalid dimensions and/or scalar type for file. Aborting!"
        
        alert.runModal()
        HideProgress()
    }
    
    func loadVolumeSet8(datFile: URL, width: Int32, height: Int32, depth: Int32) {
        
        progressStatus.stringValue = "Loading \(datFile.lastPathComponent)..."
        
        var data: Data?
        do{
            data = try Data.init(contentsOf: datFile)
        } catch {
            print(error)
        }
        
        if ( width * height * depth != data!.count) {
            ShowAlert()
            return
        }
        
        self.renderer.setVolumeData8(data: data!, dim: vector_int3(width, height, depth))
//
//        let brickDim = 128
//
//        var maxDim = max(width, height, depth)
//
//        maxDim = Int(pow(2.0, ceil(log(Double(maxDim))/log(2))))
//
//        let bricksPerDim = maxDim / brickDim
//
//        let boxDim = Double(maxDim - 1)
//        let bBox = Box(boxMin: vector3(0,0,0), boxMax: vector3(boxDim, boxDim, boxDim))
//        let ocTree: Octree<Brick<UInt8>, UInt8> = Octree<Brick<UInt8>, UInt8>(boundingBox: bBox, minimumCellSize: Double(brickDim))
//
//        let brickCenter = Double(brickDim / 2)
//        var bricks: [Brick<UInt8>] = []
//
//        progressBar.doubleValue = 0
//        let bricksCount = pow(Double(bricksPerDim), Double(3))
//        progressBar.maxValue = Double( bricksCount + 3)
//        progressStatus.stringValue = "Extracting brick 1 of \(Int(bricksCount))..."
//
//        DispatchQueue.global(qos: .userInitiated).async {
//            let start = CFAbsoluteTimeGetCurrent()
//
//            for xStep in 0..<bricksPerDim {
//                for yStep in 0..<bricksPerDim {
//                    for zStep in 0..<bricksPerDim {
//                        let xCenter = Double(xStep * brickDim) + brickCenter
//                        let yCenter = Double(yStep * brickDim) + brickCenter
//                        let zCenter = Double(zStep * brickDim) + brickCenter
//                        let brick = Brick<UInt8>(dataset: data, datasetSize: vector3(Double(width), Double(height), Double(depth)), position: vector3(xCenter, yCenter, zCenter), size: brickDim)
//                        bricks.append(brick)
//                        DispatchQueue.main.async {
//                            self.progressBar.doubleValue += 1
//                            self.progressStatus.stringValue = String(format:"Extracting brick %.0f of %.0f...", self.progressBar.doubleValue, bricksCount)
//                        }
//                    }
//                }
//            }
//
//            DispatchQueue.main.async {
//                self.progressStatus.stringValue = "Adding bricks to Octree..."
//            }
//            for idx in 0..<bricks.count {
//                ocTree.add(bricks[idx], at: bricks[idx].getPosition())
//            }
//            DispatchQueue.main.async {
//                self.progressBar.doubleValue += 1
//            }
//
//            DispatchQueue.main.async {
//                self.progressStatus.stringValue = "Generating brick LODs..."
//            }
//            ocTree.generateLODs()
//            DispatchQueue.main.async {
//                self.progressBar.doubleValue += 1
//            }
//
//            let end = CFAbsoluteTimeGetCurrent()
//            print("Took \(end-start) seconds")
//
////            DispatchQueue.main.async {
////                self.progressStatus.stringValue = "Exporting bricks..."
////            }
////
////            ocTree.exportBricks(folder: URL(fileURLWithPath: datFile.path).deletingLastPathComponent())
//
//            DispatchQueue.main.async {
//                self.HideProgress()
//            }
//
//            //print(ocTree)
//
//            self.renderer.setOctreeUInt8(ocTree: ocTree)
//        }
    }
}
