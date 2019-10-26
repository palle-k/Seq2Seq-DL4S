//
//  Util.swift
//  NMTSwift
//
//  Created by Palle Klewitz on 28.04.19.
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

public extension Sequence {
    func collect<Collector>(_ collect: (Self) throws -> Collector) rethrows -> Collector {
        return try collect(self)
    }
}

public extension Sequence {
    func top(count: Int, by comparator: (Element, Element) -> Bool) -> [Element] {
        // return sorted(by: comparator).reversed().prefix(count).collect(Array.init)
        return reduce(into: []) { acc, value in
            var i = acc.count - 1
            while i >= 0 {
                if comparator(acc[i], value) {
                    i -= 1
                } else {
                    break
                }
            }
            if i < count - 1 {
                acc.insert(value, at: i + 1)
                
                if acc.count > count {
                    acc.removeLast()
                }
            }
        }
    }
}


public extension Collection {
    subscript(safe index: Index) -> Element? {
        return indices.contains(index) ? self[index] : nil
    }
}
