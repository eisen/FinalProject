//
//  IsoFeatures.swift
//  Final Project
//
//  Created by Nishita Kharche & Eisen Montalvo on 4/20/22.
//

import Foundation
import Cocoa
import PythonKit

public class IsoFeatures {
    private static var isovalues: [Int] = []
    public static var done: Bool = false

    static func GetIsoValues(url: String, dim: vector_int3, scalarSize: ScalarSize) {
        let rootPath = Bundle.main.bundlePath + "/Contents/Resources"

        let sys = Python.import("sys")

        sys.path.append(rootPath)

        let pythonIso = Python.import("IsoFeatures")

        IsoFeatures.done = false
        DispatchQueue.global(qos: .userInitiated).async {
            let isovalues = pythonIso.getRepresentativeIsosurfaces(url, dim.x, dim.y, dim.z, scalarSize.rawValue)

            DispatchQueue.main.async {
                IsoFeatures.isovalues = isovalues.description.replacingOccurrences(of: "[ ", with: "").replacingOccurrences(of: "]", with: "").replacingOccurrences(of: "  ", with: " ").components(separatedBy: " ").map { Int($0)! }
                IsoFeatures.done = true
            }
            
        }
    }
}
