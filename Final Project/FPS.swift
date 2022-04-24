//
//  FPS.swift
//  Final Project
//
//  Created by Nishita Kharche & Eisen Montalvo on 4/11/22.
//

import Foundation

class FPS: NSObject
{
    let UpdateInterval: Double!
    var accum: Double = 0
    var frames: Int = 0
    var timeLeft: Double = 0
    var previousTime: Double = 0
    
    init(interval: Double) {
        self.UpdateInterval = interval;
        self.timeLeft = UpdateInterval;
    }
    
    public func getFPS(time: Double) -> Float {
        var fps: Float = 0
        let deltaTime: Double = time - previousTime
        previousTime = time

        timeLeft -= deltaTime
        accum += 1 / deltaTime
        frames += 1

        if(timeLeft <= 0.0)
        {
            fps = Float(accum/Double(frames))

            timeLeft = UpdateInterval;
            accum = 0;
            frames = 0;
         }
        
        return fps
    }
}
