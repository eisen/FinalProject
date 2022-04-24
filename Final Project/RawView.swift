//
//  RawView.swift
//  Final Project
//
//  Created by Nishita Kharche & Eisen Montalvo on 4/11/22.
//

import Foundation
import Cocoa

class RawView: NSView {
    @IBOutlet var width: NSTextField!
    
    @IBOutlet var height: NSTextField!
    
    @IBOutlet var depth: NSTextField!
    
    @IBOutlet var byte: NSButton!
    @IBOutlet var word: NSButton!
    
    @IBOutlet var containerView: NSView!
    
    var scalarSize: ScalarSize = .BITS_8
    
    override init(frame: NSRect) {
        super.init(frame: frame)
        initSubviews()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        initSubviews()
    }
    
    public func GetScalarSize() -> ScalarSize {
        return scalarSize
    }
    
    public func SetScalarSize(size: ScalarSize) {
        scalarSize = size
        switch scalarSize {
        case .BITS_8:
            byte.state = .on
        case .BITS_16:
            word.state = .on
        }
    }
    
    public func GetWidth() -> Int {
        return width.integerValue
    }
    
    public func GetHeight() -> Int {
        return height.integerValue
    }
    
    public func GetDepth() -> Int {
        return depth.integerValue
    }
    
    private func initSubviews() {
        let nib = NSNib(nibNamed: String(describing: type(of: self)), bundle: Bundle(for: type(of: self)))
        
        nib?.instantiate(withOwner: self, topLevelObjects: nil)
        
        containerView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(containerView)
        self.addConstraints()
    }
    
    private func addConstraints() {
        NSLayoutConstraint.activate([
        self.topAnchor.constraint(equalTo: containerView.topAnchor),
        self.leadingAnchor.constraint(equalTo: containerView.leadingAnchor),
        self.trailingAnchor.constraint(equalTo: containerView.trailingAnchor),
        self.bottomAnchor.constraint(equalTo: containerView.bottomAnchor)])
    }
    
    @IBAction func ChangeScalarSize(_ sender: NSButton) {
        if sender.title == "8 bits" {
            scalarSize = .BITS_8
        } else {
            scalarSize = .BITS_16
        }
    }
}
