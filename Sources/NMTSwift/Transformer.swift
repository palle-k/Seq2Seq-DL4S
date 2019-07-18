//
//  Transformer.swift
//  NMTSwift
//
//  Created by Palle Klewitz on 08.05.19.
//

import Foundation
import DL4S


class PositionalEncoder<Element: NumericType, Device: DeviceType>: Layer {
    typealias Input = Element
    var parameters: [Tensor<Element, Device>] {
        return []
    }
    var isTrainable: Bool {
        get {
            return false
        }
        set {
            // noop
        }
    }
    
    func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        let x = inputs[0]
        
        let seqlen = x.shape[0]
        let hiddenSize = x.shape[1]
        
        let ramp = Tensor<Element, Device>.arange(lowerBound: 0, upperBound: 1, by: 2 / Element(hiddenSize))
        let pramp = ramp.exp(withBase: 10000)
        let pos = Tensor<Element, Device>.arange(lowerBound: 0, upperBound: Element(seqlen), by: 1).view(as: -1, 1)
        
        let sinEnc = sin(pos / pramp).unsqueeze(at: 2)
        let cosEnc = cos(pos / pramp).unsqueeze(at: 2)
        let stacked = stack(sinEnc, cosEnc, axis: 2)
        
        return x + stacked.view(as: seqlen, hiddenSize)
    }
}

class MultiHeadAttention<Element: RandomizableType, Device: DeviceType>: Layer {
    typealias Input = Element
    
    var isTrainable: Bool = true
    
    let queryLinear: Dense<Element, Device>
    let keyLinear: Dense<Element, Device>
    let valuesLinear: Dense<Element, Device>
    let out: Dense<Element, Device>
    let softmax = Softmax<Element, Device>()
    
    let headCount: Int
    let size: Int
    
    var parameters: [Tensor<Element, Device>] {
        return Array([
            queryLinear.parameters,
            keyLinear.parameters,
            valuesLinear.parameters,
            out.parameters
        ].joined())
    }
    
    var trainableParameters: [Tensor<Element, Device>] {
        guard isTrainable else {
            return []
        }
        return Array([
            queryLinear.trainableParameters,
            keyLinear.trainableParameters,
            valuesLinear.trainableParameters,
            out.trainableParameters
        ].joined())
    }
    
    init(size: Int, headCount: Int) {
        queryLinear = Dense(inputFeatures: size, outputFeatures: size)
        keyLinear = Dense(inputFeatures: size, outputFeatures: size)
        valuesLinear = Dense(inputFeatures: size, outputFeatures: size)
        out = Dense(inputFeatures: size, outputFeatures: size)
        
        self.size = size
        self.headCount = headCount
    }
    
    func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        var q = inputs[0]
        var k = inputs[1]
        var v = inputs[2]
        
        q = queryLinear(q)
            .view(as: -1, headCount, size)  // [_, headCount, size]
            .permuted(to: 1, 0, 2) // [headCount, _, size]
        
        k = keyLinear(k)
            .view(as: -1, headCount, size)
            .permuted(to: 1, 0, 2) // [headCount, _, size]
        
        v = valuesLinear(v)
            .view(as: -1, headCount, size)
            .permuted(to: 1, 0, 2) // [headCount, _, size]
        
        var s = q.mmul(k.permuted(to: 0, 2, 1)) // [headCount, _, size] @ [headCount, size, _]
        s = softmax(s)
        s = s.mmul(v)
        s = s.permuted(to: 1, 0, 2)
        s = s.view(as: -1, size)
        
        return out(s)
    }
}
