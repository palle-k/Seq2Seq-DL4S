//
//  Transformer.swift
//  NMTSwift
//
//  Created by Palle Klewitz on 20.10.19.
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


public extension Tensor {
    func gaussianErrorLinear() -> Self {
        self * (self * 1.702).sigmoid()
    }
}

public func gelu<Element, Device>(_ tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
    tensor.gaussianErrorLinear()
}

// Adapted from tensorflow/swift-models

public struct FeedForwardBlock<Element: RandomizableType, Device: DeviceType>: LayerType {
    var dense1: Dense<Element, Device>
    var dense2: Dense<Element, Device>
    var dropout: Dropout<Element, Device>
    
    public var parameters: [Tensor<Element, Device>] {
        Array([dense1.parameters, dense2.parameters].joined())
    }
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {
        Array([
            dense1.parameterPaths.map((\Self.dense1).appending(path:)),
            dense2.parameterPaths.map((\Self.dense2).appending(path:))
        ].joined())
    }
    
    public var isDropoutActive: Bool {
        get {dropout.isActive}
        set {dropout.isActive = newValue}
    }
    
    public init(size: Int, hiddenSize: Int, dropoutRate: Float) {
        dense1 = Dense(inputSize: size, outputSize: hiddenSize)
        dense2 = Dense(inputSize: hiddenSize, outputSize: size)
        dropout = Dropout(rate: dropoutRate)
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        // inputs: [batchSize, timeSteps, size]
        
        let tmp1 = dense1(inputs.view(as: -1, inputs.shape[2]))
        let tmp2 = gelu(tmp1)
        let tmp3 = dropout(tmp2)
        let tmp4 = dense2(tmp3)
        return tmp4.view(as: inputs.shape[0], inputs.shape[1], -1)
    }
}

public struct AttentionInput<Element: NumericType, Device: DeviceType> {
    var key: Tensor<Element, Device>
    var value: Tensor<Element, Device>
    var query: Tensor<Element, Device>
}

public struct Attention<Element: RandomizableType, Device: DeviceType>: LayerType {
    public var parameters: [Tensor<Element, Device>] {[]}
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    
    var dropout: Dropout<Element, Device>
    var scale: Tensor<Element, Device>
    let isCausal: Bool
    
    public var isDropoutActive: Bool {
        get {dropout.isActive}
        set {dropout.isActive = newValue}
    }
    
    public init(size: Int, dropoutRate: Float, isCausal: Bool) {
        scale = Tensor(repeating: Element(size).sqrt(), shape: [])
        dropout = Dropout(rate: dropoutRate)
        self.isCausal = isCausal
    }
    
    public func callAsFunction(_ inputs: AttentionInput<Element, Device>) -> Tensor<Element, Device> {
        // inputs: [batchSize, timeSteps, size]
        let tmp1 = inputs.query.broadcastMatrixMultiplied(with: inputs.key, transposeOther: true) / scale
        
        let tmp2: Tensor<Element, Device>
        if isCausal {
            let (queryTimeSteps, keyTimeSteps) = (tmp1.shape[1], tmp1.shape[2])
            let mask = Tensor<Element, Device>(repeating: 1, shape: [queryTimeSteps, keyTimeSteps])
                .bandMatrix(belowDiagonal: nil, aboveDiagonal: queryTimeSteps - keyTimeSteps)
                .unsqueezed(at: 0)
            
            tmp2 = tmp1 * mask - (1 - mask) * 1e-10
        } else {
            tmp2 = tmp1
        }
        let tmp3 = softmax(tmp2, axis: 2)
        let tmp4 = dropout(tmp3)
        
        let tmp5 = tmp4.broadcastMatrixMultiplied(with: inputs.value)
        return tmp5
    }
}

public struct MultiHeadAttention<Element: RandomizableType, Device: DeviceType>: LayerType {
    public var parameters: [Tensor<Element, Device>] {
        Array([attention.parameters, inputTransform.parameters, outputTransform.parameters].joined())
    }
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {
        Array([
            attention.parameterPaths.map((\Self.attention).appending(path:)),
            inputTransform.parameterPaths.map((\Self.inputTransform).appending(path:)),
            outputTransform.parameterPaths.map((\Self.outputTransform).appending(path:))
        ].joined())
    }
    
    var attention: Attention<Element, Device>
    var inputTransform: Dense<Element, Device>
    var outputTransform: Dense<Element, Device>
    
    public let headCount: Int
    
    public var isDropoutActive: Bool {
        get {attention.isDropoutActive}
        set {attention.isDropoutActive = newValue}
    }
    
    public init(size: Int, headCount: Int, dropoutRate: Float, isCausal: Bool) {
        self.attention = Attention(size: size, dropoutRate: dropoutRate, isCausal: isCausal)
        self.inputTransform = Dense(inputSize: size, outputSize: size * 3)
        self.outputTransform = Dense(inputSize: size, outputSize: size)
        self.headCount = headCount
    }
    
    func splitHeads(_ input: Tensor<Element, Device>, headCount: Int) -> Tensor<Element, Device> {
        let (batchSize, timeSteps, features) = (input.shape[0], input.shape[1], input.shape[2])
        let featuresPerHead = features / headCount
        let splitLastDim = input.view(as: [batchSize, timeSteps, headCount, featuresPerHead])
        let movedToFront = splitLastDim.permuted(to: [0, 2, 1, 3])
        return movedToFront.view(as: [batchSize * headCount, timeSteps, featuresPerHead])
    }

    func joinHeads(_ input: Tensor<Element, Device>, headCount: Int) -> Tensor<Element, Device> {
        let (generalizedBatch, timeSteps, featuresPerHead) = (
            input.shape[0], input.shape[1], input.shape[2]
        )
        let batchSize = generalizedBatch / headCount
        let features = featuresPerHead * headCount
        let splitFirstDim = input.view(as: [batchSize, headCount, timeSteps, featuresPerHead])
        let movedToBack = splitFirstDim.permuted(to: [0, 2, 1, 3])
        return movedToBack.view(as: [batchSize, timeSteps, features])
    }
    
    func splitQKV(_ input: Tensor<Element, Device>) -> AttentionInput<Element, Device> {
        let featuresPerHead = input.shape[2] / 3
        let query = input[nil, nil, 0 ..< featuresPerHead]
        let key = input[nil, nil, featuresPerHead ..< 2 * featuresPerHead]
        let value = input[nil, nil, 2 * featuresPerHead ..< 3 * featuresPerHead]
        
        return AttentionInput(key: key, value: value, query: query)
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        let tmp1 = inputTransform(inputs.view(as: [-1, inputs.shape[2]])).view(as: [inputs.shape[0], inputs.shape[1], -1])
        let split = splitHeads(tmp1, headCount: headCount)
        let attentionInput = splitQKV(split)
        let attentionOutputs = attention(attentionInput)
        let joined = joinHeads(attentionOutputs, headCount: headCount)
        let output = outputTransform(joined.view(as: [-1, joined.shape[2]])).view(as: [joined.shape[0], joined.shape[1], -1])
        return output
    }
}

public struct TransformerLayer<Element: RandomizableType, Device: DeviceType> {
    public var parameters: [Tensor<Element, Device>] {
        Array([
            selfAttention.parameters,
            selfAttentionDropout.parameters,
            selfAttentionNorm.parameters,
            feedForward.parameters,
            feedForwardDropout.parameters,
            feedForwardNorm.parameters
        ].joined())
    }
    
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {
        Array([
            selfAttention.parameterPaths.map((\Self.selfAttention).appending(path:)),
            selfAttentionDropout.parameterPaths.map((\Self.selfAttentionDropout).appending(path:)),
            selfAttentionNorm.parameterPaths.map((\Self.selfAttentionNorm).appending(path:)),
            feedForward.parameterPaths.map((\Self.feedForward).appending(path:)),
            feedForwardDropout.parameterPaths.map((\Self.feedForwardDropout).appending(path:)),
            feedForwardNorm.parameterPaths.map((\Self.feedForwardNorm).appending(path:))
        ].joined())
    }
    
    var selfAttention: MultiHeadAttention<Element, Device>
    var selfAttentionDropout: Dropout<Element, Device>
    var selfAttentionNorm: LayerNorm<Element, Device>
    var feedForward: FeedForwardBlock<Element, Device>
    var feedForwardDropout: Dropout<Element, Device>
    var feedForwardNorm: LayerNorm<Element, Device>
    
    public var isDropoutActive: Bool {
        get {
            selfAttention.isDropoutActive
        }
        set {
            selfAttention.isDropoutActive = newValue
            selfAttentionDropout.isActive = newValue
            feedForward.isDropoutActive = newValue
            feedForwardDropout.isActive = newValue
        }
    }

    public init(size: Int, headCount: Int, dropoutRate: Float, isCausal: Bool) {
        selfAttention = MultiHeadAttention(
            size: size,
            headCount: headCount,
            dropoutRate: dropoutRate,
            isCausal: isCausal
        )
        selfAttentionDropout = Dropout(rate: dropoutRate)
        selfAttentionNorm = LayerNorm(inputSize: [size])
        feedForward = FeedForwardBlock(size: size, hiddenSize: 4 * size, dropoutRate: dropoutRate)
        feedForwardDropout = Dropout(rate: dropoutRate)
        feedForwardNorm = LayerNorm(inputSize: [size])
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        let tmp1 = selfAttentionNorm(inputs.view(as: -1, inputs.shape[2])).view(as: inputs.shape)
        let tmp2 = selfAttention(tmp1)
        let tmp3 = selfAttentionDropout(tmp2)
        let attended = inputs + tmp3
        
        let tmp4 = feedForwardNorm(attended)
        let tmp5 = feedForward(tmp4)
        let tmp6 = feedForwardDropout(tmp5)
        
        let result = attended + tmp6
        return result
    }
}

public struct TransformerBase<Element: RandomizableType, Device: DeviceType>: LayerType {
    public var parameters: [Tensor<Element, Device>] {
        embedding.parameters + transformerLayers.flatMap {$0.parameters}
    }
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {
        Array([
            embedding.parameterPaths.map((\Self.embedding).appending(path:)),
            transformerLayers.enumerated().flatMap { idx, layer in
                layer.parameterPaths.map((\Self.transformerLayers[idx]).appending(path:))
            },
        ].joined())
    }
    
    public var isDropoutActive: Bool {
        get {transformerLayers.first?.isDropoutActive ?? false}
        set {
            for i in transformerLayers.indices {
                transformerLayers[i].isDropoutActive = newValue
            }
        }
    }
    
    var embedding: Embedding<Element, Device>
    var transformerLayers: [TransformerLayer<Element, Device>]
    
    public init(vocabularySize: Int, layerSizes: [Int], headCounts: [Int], dropoutRate: Float, isCausal: Bool) {
        guard let embeddingSize = layerSizes.first else {
            fatalError("At least one layer must be given")
        }
        embedding = Embedding(inputFeatures: vocabularySize, outputSize: embeddingSize)
        transformerLayers = zip(layerSizes, headCounts).map {
            TransformerLayer(size: $0, headCount: $1, dropoutRate: dropoutRate, isCausal: isCausal)
        }
    }
    
    public func callAsFunction(_ inputs: Tensor<Int32, Device>) -> Tensor<Element, Device> {
        let flat = inputs.flattened()
        let emb = embedding(flat).view(as: inputs.shape + [-1])
        let transformed = transformerLayers.reduce(emb, {$1.callAsFunction($0)})
        return transformed
    }
}

public struct TransformerLanguageModel<Element: RandomizableType, Device: DeviceType>: LayerType {
    public var parameters: [Tensor<Element, Device>] {
        encoder.parameters + layerNorm.parameters
    }
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {
        Array([
            encoder.parameterPaths.map((\Self.encoder).appending(path:)),
            layerNorm.parameterPaths.map((\Self.layerNorm).appending(path:))
        ].joined())
    }
    
    var encoder: TransformerBase<Element, Device>
    var layerNorm: LayerNorm<Element, Device>
    
    public init(vocabularySize: Int, layerSizes: [Int], headCounts: [Int], dropoutRate: Float) {
        guard let lastSize = layerSizes.last else {
            fatalError("At least one layer must be given")
        }
        encoder = TransformerBase(vocabularySize: vocabularySize, layerSizes: layerSizes, headCounts: headCounts, dropoutRate: dropoutRate, isCausal: false)
        layerNorm = LayerNorm(inputSize: [lastSize])
    }
    
    public func callAsFunction(_ inputs: Tensor<Int32, Device>) -> Tensor<Element, Device> {
        let transformed = encoder(inputs)
        let normalized = layerNorm(transformed.view(as: -1, transformed.shape[2]))
        let deembedded = encoder.embedding.embeddingMatrix
            .broadcastMatrixMultiplied(with: normalized, transposeSelf: true, transposeOther: false)
            .view(as: [transformed.shape[0], transformed.shape[1], -1])
        return softmax(deembedded, axis: 2)
    }
}
