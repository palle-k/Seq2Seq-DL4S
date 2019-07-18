//
//  Visualizer.swift
//  NMTSwift
//
//  Created by Palle Klewitz on 27.04.19.
//  Copyright (c) 2019 Palle Klewitz
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

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
