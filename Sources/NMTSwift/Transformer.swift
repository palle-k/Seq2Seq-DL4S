//
//  Transformer.swift
//  NMTSwift
//
//  Created by Palle Klewitz on 20.10.19.
//

import Foundation
import DL4S


struct PositionalEncoder<Element: NumericType, Device: DeviceType>: LayerType {
    var parameters: [Tensor<Element, Device>] {[]}
    var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    
    var modelDim: Int
    
    func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        var x = inputs
        x = x * sqrt(Tensor(Element(modelDim)))
        
        let seqlen = inputs.shape[1]
        
        let expVals = Tensor(linearRampWithLowerBound: 0, upperBound: Element(seqlen), by: 1)
        .view(as: -1, 1) /
        Tensor(10000)
        .raised(toPowerOf:
            (2 * Tensor<Element, Device>(linearRampWithLowerBound: Element(0), upperBound: Element(modelDim / 2), by: Element(1)))
        )
        .view(as: 1, -1)
        
        let encoding = Tensor(stacking: [
            sin(expVals), cos(expVals)
        ], along: 1)
        
        return x + encoding
    }
}


struct MultiHeadAttention<Element: RandomizableType, Device: DeviceType>: LayerType {
    var parameters: [Tensor<Element, Device>] {
        Array([
            makeQ.parameters,
            makeV.parameters,
            makeK.parameters,
            output.parameters,
            dropout.parameters
        ].joined())
    }
    
    var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {
        Array([
            makeQ.parameterPaths.map((\Self.makeQ).appending(path:)),
            makeV.parameterPaths.map((\Self.makeV).appending(path:)),
            makeK.parameterPaths.map((\Self.makeK).appending(path:)),
            output.parameterPaths.map((\Self.output).appending(path:)),
            dropout.parameterPaths.map((\Self.dropout).appending(path:))
        ].joined())
    }
    
    let heads: Int
    let modelDim: Int
    
    var makeQ: Dense<Element, Device>
    var makeV: Dense<Element, Device>
    var makeK: Dense<Element, Device>
    var output: Dense<Element, Device>
    
    var dropout: Dropout<Element, Device>
    
    init(heads: Int, modelDim: Int, dropout: Float = 0.1) {
        self.heads = heads
        self.modelDim = modelDim
        
        self.makeQ = Dense(inputSize: modelDim, outputSize: modelDim)
        self.makeV = Dense(inputSize: modelDim, outputSize: modelDim)
        self.makeK = Dense(inputSize: modelDim, outputSize: modelDim)
        self.output = Dense(inputSize: modelDim, outputSize: modelDim)
        self.dropout = Dropout(rate: dropout)
    }
    
    func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        let batchSize = inputs.shape[0]
        
        fatalError()
    }
}
