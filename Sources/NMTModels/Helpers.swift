//
//  Helpers.swift
//  NMTSwift
//
//  Created by Palle Klewitz on 14.03.19.
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


public struct Helper {
    public static func decodingLoss<Element: RandomizableType, Device: DeviceType>(forExpectedSequence expectedSequence: [Int32], actualSequence: Tensor<Element, Device>) -> Tensor<Element, Device> {
        let subsequence = Array(expectedSequence.prefix(actualSequence.shape[0]))
        let loss = categoricalCrossEntropy(expected: Tensor(subsequence), actual: actualSequence.squeezed(at: 1)[0 ..< subsequence.count])
        
        return loss
    }
    
    public static func sequence<Element: RandomizableType, Device: DeviceType>(from tensor: Tensor<Element, Device>) -> [Int32] {
        var s: [Int32] = []
        
        for i in 0 ..< tensor.shape[0] {
            s.append(Int32(tensor[i].argmax()))
        }
        
        return s
    }
}

fileprivate extension Array where Element: Comparable {
    mutating func insertOrdered(_ element: Element) -> Int {
        if let idx = self.firstIndex(where: {$0 > element}) {
            self.insert(element, at: idx)
            return idx
        } else {
            self.append(element)
            return self.count - 1
        }
    }
}
