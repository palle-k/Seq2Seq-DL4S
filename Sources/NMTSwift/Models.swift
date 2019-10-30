//
//  Encoder.swift
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

struct Encoder<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {
        embedding.parameterPaths.map((\Self.embedding).appending(path:)) +
        rnn.parameterPaths.map((\Self.rnn).appending(path:))
    }

    var parameters: [Tensor<Element, Device>] {
        return embedding.parameters + rnn.parameters
    }
    
    private var embedding: Embedding<Element, Device>
    private var rnn: GRU<Element, Device>
    
    init(inputSize: Int, hiddenSize: Int) {
        self.embedding = Embedding(inputFeatures: inputSize, outputSize: hiddenSize)
        self.rnn = GRU(inputSize: hiddenSize, hiddenSize: hiddenSize)
    }
    
    func callAsFunction(_ inputs: Tensor<Int32, Device>) -> Tensor<Element, Device> {
        precondition(inputs.count == 1)
        
        let batchSize = 1
        let length = inputs.shape[0]
        
        let embedded = self.embedding(inputs)
        let rnnIn = embedded.view(as: length, batchSize, -1)
        
        let (_, rnnOut) = self.rnn(rnnIn)
        
        return rnnOut()
    }
}

struct BidirectionalEncoder<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {
        embedding.parameterPaths.map((\Self.embedding).appending(path:)) +
        rnn.parameterPaths.map((\Self.rnn).appending(path:))
    }

    var parameters: [Tensor<Element, Device>] {
        return embedding.parameters + rnn.parameters
    }
    
    var embedding: Embedding<Element, Device>
    var rnn: Bidirectional<GRU<Element, Device>>
    
    init(inputSize: Int, embeddingSize: Int, hiddenSize: Int, embeddingsFile: URL?, words: [String]) {
        if let embeddingsFile = embeddingsFile, let embedding = Embedding<Element, Device>(words: words, embeddingsURL: embeddingsFile, verbose: true) {
            self.embedding = embedding
            print("Using pretrained embeddings.")
        } else {
            self.embedding = Embedding(inputFeatures: inputSize, outputSize: embeddingSize)
        }
        self.rnn = Bidirectional(
            forward: GRU(inputSize: embeddingSize, hiddenSize: hiddenSize / 2, direction: .forward),
            backward: GRU(inputSize: embeddingSize, hiddenSize: hiddenSize / 2, direction: .backward)
        )
    }
    
    func callAsFunction(_ inputs: Tensor<Int32, Device>) -> Tensor<Element, Device> {
        let batchSize = 1
        let length = inputs.shape[0]
        
        let embedded = self.embedding(inputs)
        let rnnIn = embedded.view(as: length, batchSize, -1)
        
        let ((_, forwardOut), (_, backwardOut)) = self.rnn(rnnIn)
        
        return stack([forwardOut(), backwardOut()], along: 2)
    }
}

struct GeneralAttention<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    var W_a: Tensor<Element, Device>
    
    let encoderHiddenSize: Int
    let decoderHiddenSize: Int
    
    let isTemporal: Bool
    
    var parameters: [Tensor<Element, Device>] {
        return [W_a]
    }
    
    var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[\Self.W_a]}
    
    init(encoderHiddenSize: Int, decoderHiddenSize: Int, temporal: Bool) {
        self.encoderHiddenSize = encoderHiddenSize
        self.decoderHiddenSize = decoderHiddenSize
        self.isTemporal = temporal
        
        W_a = Tensor(normalDistributedWithShape: [decoderHiddenSize, encoderHiddenSize], mean: 0, stdev: Element(1 / sqrt(Float(encoderHiddenSize))), requiresGradient: true)
        #if DEBUG
        W_a.tag = "W_a"
        #endif
    }
    
    func callAsFunction(_ inputs: (Tensor<Element, Device>, Tensor<Element, Device>, Tensor<Element, Device>)) -> Tensor<Element, Device> {
        let encoderStateSequence = inputs.0 // [seqlen, batchSize, encHS]
        let decoderState = inputs.1 // [batchSize, decHS]
        
        // let seqlen = encoderStateSequence.shape[0]
        let batchSize = decoderState.shape[0]
        
        let flatEnc = encoderStateSequence.view(as: -1, encoderHiddenSize) // [seqlen*batchSize, encHS]
        let encMulW = flatEnc.matrixMultiplied(with: W_a) // [seqlen*batchSize, decHS]
        
        let scores = encMulW
            .matrixMultiplied(with: decoderState.view(as: -1, 1)) // [seqlen*batchSize, 1]
            .view(as: [-1, batchSize]) // [seqlen, batchSize]
        
        if !isTemporal  {
            return softmax(scores.T).T
        }
        
        let attnHistory = inputs.2 // [seqlen, batchSize]
        if attnHistory.shape[0] == 0 {
            return softmax(scores.T).T
        }
        
        let expScores = exp(scores) // [seqlen, batchSize]
        return expScores / attnHistory
    }
}

struct TanhAttention<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    var parameters: [Tensor<Element, Device>] {
        return [W_h, W_s, b, v]
    }
    
    var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[
        \Self.W_h, \Self.W_s, \Self.b, \Self.v
    ]}
    
    var W_h: Tensor<Element, Device>
    var W_s: Tensor<Element, Device>
    var b: Tensor<Element, Device>
    var v: Tensor<Element, Device>
    
    let encoderHiddenSize: Int
    let decoderHiddenSize: Int
    let latentSize: Int
    
    let isTemporal: Bool
    
    init(encoderHiddenSize: Int, decoderHiddenSize: Int, latentSize: Int, temporal: Bool) {
        self.encoderHiddenSize = encoderHiddenSize
        self.decoderHiddenSize = decoderHiddenSize
        self.latentSize = latentSize
        self.isTemporal = temporal
        
        W_h = Tensor<Element, Device>(normalDistributedWithShape: decoderHiddenSize, latentSize, mean: 0, stdev: Element(1 / sqrt(Float(decoderHiddenSize))), requiresGradient: true)
        W_s = Tensor<Element, Device>(normalDistributedWithShape: encoderHiddenSize, latentSize, mean: 0, stdev: Element(1 / sqrt(Float(encoderHiddenSize))), requiresGradient: true)
        b = Tensor<Element, Device>(normalDistributedWithShape: latentSize, mean: 0, stdev: Element(1 / sqrt(Float(latentSize))), requiresGradient: true)
        v = Tensor<Element, Device>(normalDistributedWithShape: latentSize, mean: 0, stdev: Element(1 / sqrt(Float(latentSize))), requiresGradient: true)
    }
    
    func callAsFunction(_ inputs: (encoderStateSequence: Tensor<Element, Device>, decoderState: Tensor<Element, Device>, attentionHistory: Tensor<Element, Device>)) -> (Tensor<Element, Device>, Tensor<Element, Device>) {
        let encoderStateSequence = inputs.0 // [seqlen, batchSize, encHS]
        let decoderState = inputs.1 // [batchSize, decHS]
        
        // let seqlen = encoderStateSequence.shape[0]
        let batchSize = decoderState.shape[0]
        
        let encIn = encoderStateSequence.view(as: -1, encoderHiddenSize)
        
        let encScore = encIn.matrixMultiplied(with: W_s).view(as: [-1, batchSize, latentSize]) // [seqlen * batchSize, latentSize]
        let decScore = decoderState.matrixMultiplied(with: W_h) // [batchSize, latentSize]
        
        let scores = tanh(encScore + decScore + b) // [seqlen, batchSize, latentSize]
            .view(as: -1, latentSize) // [seqlen * batchSize, latentSize]
            .matrixMultiplied(with: v.view(as: -1, 1)) // [seqlen * batchSize, 1]
            .view(as: -1, batchSize) // [seqlen, batchSize]
        
        if !isTemporal  {
            return (softmax(scores.T).T, scores)
        }
        
        let expScores = exp(scores) // [seqlen, batchSize]
        let attnHistory = inputs.attentionHistory // [seqlen, batchSize]
        
        if attnHistory.dim == 0 {
            return (softmax(scores.T).T, expScores)
        }
        
        return (normalize(expScores / attnHistory, along: 0), expScores + attnHistory)
    }
}

func normalize<Element, Device>(_ tensor: Tensor<Element, Device>, along axis: Int) -> Tensor<Element, Device> {
    return tensor / tensor.reduceSum(along: axis).unsqueezed(at: axis)
}

struct AttentionCombine<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    var parameters: [Tensor<Element, Device>] {[]}
    
    func callAsFunction(_ inputs: (Tensor<Element, Device>, Tensor<Element, Device>)) -> Tensor<Element, Device> {
        let scores = inputs.0 // [seqlen, batchsize]
        let weights = inputs.1 // [seqlen, batchSize, hiddenSize]
        
        let scoredWeights = (scores.unsqueezed(at: 2) * weights).reduceSum(along: 0) // [batchSize, hiddenSize]
        
        return scoredWeights
    }
}

struct AttentionDecoder<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    var parameters: [Tensor<Element, Device>] {
        Array([
            embed.parameters,
            attention.parameters,
            combine.parameters,
            rnn.parameters,
            deembed.parameters,
            softmax.parameters
        ].joined())
    }
    
    var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {
        Array([
            embed.parameterPaths.map((\Self.embed).appending(path:)),
            attention.parameterPaths.map((\Self.attention).appending(path:)),
            combine.parameterPaths.map((\Self.combine).appending(path:)),
            rnn.parameterPaths.map((\Self.rnn).appending(path:)),
            deembed.parameterPaths.map((\Self.deembed).appending(path:)),
            softmax.parameterPaths.map((\Self.softmax).appending(path:))
        ].joined())
    }
    
    var embed: Embedding<Element, Device>
    var attention: TanhAttention<Element, Device>
    var combine: AttentionCombine<Element, Device>
    var rnn: GRU<Element, Device>
    var deembed: Dense<Element, Device>
    var softmax: Softmax<Element, Device>
    
    init(inputSize: Int, embeddingSize: Int, hiddenSize: Int, words: [String], embeddingsFile: URL?, useTemporalAttention: Bool) {
        if let embeddingsFile = embeddingsFile, let embed = Embedding<Element, Device>(words: words, embeddingsURL: embeddingsFile, verbose: true) {
            self.embed = embed
            print("Using pretrained embeddings.")
        } else {
            self.embed = Embedding(inputFeatures: inputSize, outputSize: embeddingSize)
        }
        // attention = GeneralAttention(encoderHiddenSize: hiddenSize, decoderHiddenSize: hiddenSize).asAny()
        attention = TanhAttention(encoderHiddenSize: hiddenSize, decoderHiddenSize: hiddenSize, latentSize: hiddenSize, temporal: useTemporalAttention)
        combine = AttentionCombine()
        rnn = GRU(inputSize: hiddenSize + embeddingSize, hiddenSize: hiddenSize)
        deembed = Dense(inputSize: hiddenSize, outputSize: inputSize)
        softmax = Softmax()
    }
    
    func callAsFunction(_ inputs: (input: Tensor<Int32, Device>, state: Tensor<Element, Device>, encoderStates: Tensor<Element, Device>, attentionHistory: Tensor<Element, Device>)) -> (output: Tensor<Element, Device>, state: Tensor<Element, Device>, attentionScores: Tensor<Element, Device>, attentionHistory: Tensor<Element, Device>) {
        // input: [(1, )?batchSize] (squeezed at a later stage, so dim not important)
        // state: [batchSize, hiddenSize]
        // encoderState: [seqlen, batchSize, hiddenSize]
        
        let embedded = embed(inputs.input.view(as: -1)) // [batchSize, hiddenSize]
        
        let (attentionScores, history) = attention((inputs.encoderStates, inputs.state, inputs.attentionHistory)) // [seqlen, batchSize]
        let weightedStates = combine((attentionScores, inputs.encoderStates)) // [batchSize, hiddenSize]
        
        let rnnInput = stack([embedded, weightedStates], along: 1) // [batchSize, 2 * hiddenSize]
        let (newState, _) = rnn.callAsFunction(rnnInput.view(as: [1, 1, -1]), state: inputs.state)
        
        let deembedded = deembed(newState)
        let wordScores = softmax(deembedded)
        
        return (output: wordScores, state: newState, attentionScores: attentionScores, attentionHistory: history)
    }
}
