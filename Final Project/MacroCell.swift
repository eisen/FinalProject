//
//  MacroCell.swift
//  Final Project
//
//  Created by Eisen Montalvo on 4/8/22.
//

import Foundation

public class MacroCell<T: FixedWidthInteger>: CustomStringConvertible {
    var min_value: T = T.max
    var max_value: T = T.min
    
    var adds: Int = 0
    
    public var description: String {
        return "\(min_value)...\(max_value)"
    }
    
    public init() {
    }
    
    public func isEmpty() -> Bool {
        return self.min_value == 0 && self.max_value == 0
    }
    
    public func addScalar(scalar: T) {
        if(scalar < min_value) {
            self.min_value = scalar
        }
        
        if(scalar > max_value) {
            self.max_value = scalar
        }
        
        adds += 1
        if(adds > 512) {
            print("Added more scalars than supposed")
        }
    }
}

