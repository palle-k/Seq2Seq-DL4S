//
//  Visualizer.swift
//  Commander
//
//  Created by Palle Klewitz on 27.04.19.
//

import Foundation
import DL4S
import CoreGraphics
import QuickLook
import AppKit
import Quartz
import QuartzCore


struct AttentionVisualizer {
    func render(attention: Tensor<Float, CPU>, source: [String], result: [String], to path: String) {
        let url = URL(fileURLWithPath: path)
        guard let ctx = CGContext(url as CFURL, mediaBox: [CGRect(x: 0, y: 0, width: 800, height: 600)], nil) else {
            return
        }
        
        
        
        
        
        ctx.flush()
        
        guard let ql = QLPreviewPanel.shared() else {
            return
        }
        ql.representedURL = url
        ql.makeKeyAndOrderFront(nil)
    }
}
