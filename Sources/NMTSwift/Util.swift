//
//  Util.swift
//  NMTSwift
//
//  Created by Palle Klewitz on 26.10.19.
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
import Commander
import PythonKit
import DL4S

func validatePositive<N: Numeric & Comparable>(message: String) -> (N) throws -> N {
    return { num in
        if num >= N.zero {
            return num
        } else {
            throw NSError(domain: "NMTSwiftErrorDomain", code: 2, userInfo: [NSLocalizedDescriptionKey: message])
        }
    }
}

func validatePositive<N: Numeric & Comparable>(message: String) -> (N?) throws -> N? {
    return { n in
        guard let num = n else {
            return n
        }
        if num >= N.zero {
            return num
        } else {
            throw NSError(domain: "NMTSwiftErrorDomain", code: 2, userInfo: [NSLocalizedDescriptionKey: message])
        }
    }
}

func validateRange<N: Numeric & Comparable>(_ range: ClosedRange<N>, message: String) -> (N) throws -> N {
    return { num in
        if range ~= num {
            return num
        } else {
            throw NSError(domain: "NMTSwiftErrorDomain", code: 3, userInfo: [NSLocalizedDescriptionKey: message])
        }
    }
}

func validatePathExists(isDirectory: Bool? = nil) -> (String) throws -> String {
    return { path in
        var isActualDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: path, isDirectory: &isActualDirectory), isDirectory == nil || isDirectory == isActualDirectory.boolValue {
            return path
        } else {
            throw NSError(domain: "NMTSwiftErrorDomain", code: 1, userInfo: [NSLocalizedDescriptionKey: "Path '\(path)' does not exist."])
        }
    }
}

func validatePathExists(isDirectory: Bool? = nil) -> (String?) throws -> String? {
    return { p in
        guard let path = p else {
            return p
        }
        var isActualDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: path, isDirectory: &isActualDirectory), isDirectory == nil || isDirectory == isActualDirectory.boolValue {
            return path
        } else {
            throw NSError(domain: "NMTSwiftErrorDomain", code: 1, userInfo: [NSLocalizedDescriptionKey: "Path '\(path)' does not exist."])
        }
    }
}

private let plt = Python.import("matplotlib.pyplot")
private let np = Python.import("numpy")

enum Plot {
    static func matshow<Device>(_ tensor: Tensor<Float, Device>) {
        let arr = tensor
            .elements
            .makeNumpyArray()
            .reshape(tensor.shape)
        plt.matshow(arr)
    }
    
    static func show() {
        plt.show()
    }
    
    static func showAttention<Device>(_ tensor: Tensor<Float, Device>, source: [String], destination: [String], title: String? = nil) {
        let ticker = Python.import("matplotlib.ticker")
        
        let arr = tensor
            .elements
            .makeNumpyArray()
            .reshape(tensor.shape)
        
        let fig = plt.figure()
        let ax = fig.add_subplot(111)
        let color_axis = ax.matshow(arr)
        fig.colorbar(color_axis)
        ax.set_xticklabels([""] + source, rotation: 90)
        ax.set_yticklabels([""] + destination)
        if let title = title {
            ax.set_title(title)
        }
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        
        // plt.ion()
        plt.show()
    }
}

struct TBSummaryWriter {
    let summaryWriter: PythonObject
    
    init(runName: String) {
        let tensorboardX = Python.import("tensorboardX")
        let writer = tensorboardX.SummaryWriter(runName)
        self.summaryWriter = writer
    }
    
    func addScalar(name: String, value: Double, iteration: Int) {
        summaryWriter.add_scalar(name, value, iteration)
    }
}

extension Collection {
    func pmap<ElementOfResult>(_ transform: (Element) -> ElementOfResult) -> [ElementOfResult] {
        var result: [ElementOfResult?] = Array(repeating: nil, count: count)
        
        DispatchQueue.concurrentPerform(iterations: count) { offset in
            let idx = self.index(self.startIndex, offsetBy: offset)
            let element = self[idx]
            let mapped = transform(element)
            
            result[offset] = mapped
        }
        
        return result.compactMap {$0}
    }
}
