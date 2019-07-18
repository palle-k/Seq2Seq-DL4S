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

class Encoder<Element: RandomizableType, Device: DeviceType>: Layer, Codable {
    var isTrainable: Bool = true
    
    var parameters: [Tensor<Element, Device>] {
        return embedding.parameters + rnn.parameters
    }
    
    private let embedding: Embedding<Element, Device>
    private let rnn: GRU<Element, Device>
    
    init(inputSize: Int, hiddenSize: Int) {
        self.embedding = Embedding(inputFeatures: inputSize, outputSize: hiddenSize)
        self.rnn = GRU(inputSize: hiddenSize, hiddenSize: hiddenSize, shouldReturnFullSequence: false)
    }
    
    
    func forward(_ inputs: [Tensor<Int32, Device>]) -> Tensor<Element, Device> {
        precondition(inputs.count == 1)
        
        let batchSize = 1
        let length = inputs[0].shape[0]
        
        let embedded = self.embedding.forward(inputs[0])
        let rnnIn = embedded.view(as: length, batchSize, -1)
        
        let rnnOut = self.rnn.forward(rnnIn)
        
        return rnnOut
    }
}

class BidirectionalEncoder<Element: RandomizableType, Device: DeviceType>: Layer, Codable {
    var isTrainable: Bool = true
    
    var parameters: [Tensor<Element, Device>] {
        return embedding.parameters + rnn.parameters
    }
    
    var trainableParameters: [Tensor<Element, Device>] {
        guard isTrainable else {
            return []
        }
        return embedding.trainableParameters + rnn.trainableParameters
    }
    
    let embedding: Embedding<Element, Device>
    let rnn: Bidirectional<GRU<Element, Device>>
    
    init(inputSize: Int, embeddingSize: Int, hiddenSize: Int, embeddingsFile: URL?, words: [String]) {
        if let embeddingsFile = embeddingsFile, let embedding = Embedding<Element, Device>(words: words, embeddingsURL: embeddingsFile, verbose: true) {
            self.embedding = embedding
            print("Using pretrained embeddings.")
        } else {
            self.embedding = Embedding(inputFeatures: inputSize, outputSize: embeddingSize)
        }
        self.rnn = Bidirectional(
            forwardLayer: GRU(inputSize: embeddingSize, hiddenSize: hiddenSize / 2, direction: .forward, shouldReturnFullSequence: true),
            backwardLayer: GRU(inputSize: embeddingSize, hiddenSize: hiddenSize / 2, direction: .backward, shouldReturnFullSequence: true)
        )
    }
    
    func forward(_ inputs: [Tensor<Int32, Device>]) -> Tensor<Element, Device> {
        precondition(inputs.count == 1)
        
        let batchSize = 1
        let length = inputs[0].shape[0]
        
        let embedded = self.embedding.forward(inputs[0])
        let rnnIn = embedded.view(as: length, batchSize, -1)
        
        let rnnOut = self.rnn(rnnIn)
        
        return rnnOut
    }
}


class Decoder<Element: RandomizableType, Device: DeviceType>: Layer, Codable {
    typealias Input = Int32
    
    var isTrainable: Bool = true
    
    var parameters: [Tensor<Element, Device>] {
        return Array([
            embedding.parameters,
            rnn.parameters,
            dense.parameters,
            softmax.parameters
        ].joined())
    }
    
    var trainableParameters: [Tensor<Element, Device>] {
        return isTrainable ? Array([
            embedding.trainableParameters,
            rnn.trainableParameters,
            dense.trainableParameters,
            softmax.trainableParameters
        ].joined()) : []
    }
    
    private let embedding: Embedding<Element, Device>
    private let rnn: GRU<Element, Device>
    private let dense: Dense<Element, Device>
    private let softmax: Softmax<Element, Device>
    
    init(inputSize: Int, hiddenSize: Int) {
        self.embedding = Embedding(inputFeatures: inputSize, outputSize: hiddenSize)
        self.rnn = GRU(inputSize: hiddenSize, hiddenSize: hiddenSize, shouldReturnFullSequence: true)
        self.dense = Dense(inputFeatures: hiddenSize, outputFeatures: inputSize)
        self.softmax = Softmax()
    }
    
    func forward(_ inputs: [Tensor<Int32, Device>]) -> Tensor<Element, Device> {
        let initialHidden = Tensor<Element, Device>(repeating: 0, shape: 1, self.rnn.hiddenSize) // batchSize x hiddenSize
        return forwardFullSequence(input: inputs[0], initialHidden: initialHidden)
    }
    
    func forward(input: Tensor<Int32, Device>, previousHidden: Tensor<Element, Device>) -> (output: Tensor<Element, Device>, hidden: Tensor<Element, Device>) {
        let embedded = embedding.forward(input.view(as: -1)).view(as: 1, 1, -1)
        let nextHidden = rnn.forward(embedded, previousHidden).view(as: 1, -1)
        let deembedded = dense.forward(nextHidden)
        let probs = softmax.forward(deembedded)
        
        return (probs, nextHidden)
    }
    
    func forwardFullSequence(input: Tensor<Int32, Device>, initialHidden: Tensor<Element, Device>) -> Tensor<Element, Device> {
        let seqlen = input.shape[0]
        let embedded = embedding.forward(input.view(as: -1)).view(as: seqlen, 1, -1)
        // let rectified = relu(embedded)
        let hidden = rnn.forward(embedded, initialHidden).squeeze() // get rid of batch size dimension -> [seqlen x hiddenSize]
        let deembedded = dense.forward(hidden)
        let probs = softmax.forward(deembedded)
        
        return probs
    }
}


class GeneralAttention<Element: RandomizableType, Device: DeviceType>: Layer, Codable {
    var isTrainable: Bool = true
    
    let W_a: Tensor<Element, Device>
    let softmax = Softmax<Element, Device>()
    let encoderHiddenSize: Int
    let decoderHiddenSize: Int
    
    let isTemporal: Bool
    
    var parameters: [Tensor<Element, Device>] {
        return [W_a]
    }
    
    init(encoderHiddenSize: Int, decoderHiddenSize: Int, temporal: Bool) {
        self.encoderHiddenSize = encoderHiddenSize
        self.decoderHiddenSize = decoderHiddenSize
        self.isTemporal = temporal
        W_a = Tensor<Element, Device>(repeating: 0, shape: decoderHiddenSize, encoderHiddenSize, requiresGradient: true)
        Random.fillNormal(W_a, mean: 0, stdev: Element(1 / sqrt(Float(encoderHiddenSize))))
        
        W_a.tag = "W_a"
    }
    
    func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        let encoderStateSequence = inputs[0] // [seqlen, batchSize, encHS]
        let decoderState = inputs[1] // [batchSize, decHS]
        
        // let seqlen = encoderStateSequence.shape[0]
        let batchSize = decoderState.shape[0]
        
        let flatEnc = encoderStateSequence.view(as: -1, encoderHiddenSize) // [seqlen*batchSize, encHS]
        let encMulW = flatEnc.mmul(W_a) // [seqlen*batchSize, decHS]
        
        let scores = encMulW
            .mmul(decoderState.view(as: -1, 1)) // [seqlen*batchSize, 1]
            .view(as: -1, batchSize) // [seqlen, batchSize]
        
        if !isTemporal  {
            return softmax(scores.T).T
        }
        
        let attnHistory = inputs[2] // [seqlen, batchSize]
        if attnHistory.shape[0] == 0 {
            return softmax(scores.T).T
        }
        
        let expScores = exp(scores) // [seqlen, batchSize]
        return expScores / attnHistory
    }
}

class TanhAttention<Element: RandomizableType, Device: DeviceType>: Layer, Codable {
    var isTrainable: Bool = true
    
    let W_h: Tensor<Element, Device>
    let W_s: Tensor<Element, Device>
    let b: Tensor<Element, Device>
    let v: Tensor<Element, Device>
    
    let encoderHiddenSize: Int
    let decoderHiddenSize: Int
    let latentSize: Int
    
    let softmax = Softmax<Element, Device>()
    
    let isTemporal: Bool
    
    var parameters: [Tensor<Element, Device>] {
        return [W_h, W_s, b, v]
    }
    
    init(encoderHiddenSize: Int, decoderHiddenSize: Int, latentSize: Int, temporal: Bool) {
        self.encoderHiddenSize = encoderHiddenSize
        self.decoderHiddenSize = decoderHiddenSize
        self.latentSize = latentSize
        self.isTemporal = temporal
        
        W_h = Tensor<Element, Device>(repeating: 0, shape: decoderHiddenSize, latentSize, requiresGradient: true)
        W_s = Tensor<Element, Device>(repeating: 0, shape: encoderHiddenSize, latentSize, requiresGradient: true)
        b = Tensor<Element, Device>(repeating: 0, shape: latentSize, requiresGradient: true)
        v = Tensor<Element, Device>(repeating: 0, shape: latentSize, requiresGradient: true)
        
        Random.fillNormal(W_h, mean: 0, stdev: Element(1 / sqrt(Float(decoderHiddenSize))))
        Random.fillNormal(W_s, mean: 0, stdev: Element(1 / sqrt(Float(encoderHiddenSize))))
        Random.fillNormal(b, mean: 0, stdev: Element(1 / sqrt(Float(latentSize))))
        Random.fillNormal(v, mean: 0, stdev: Element(1 / sqrt(Float(latentSize))))
    }
    
    func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        fatalError("forward(_:) is unavailable.")
    }
    
    func forward(_ inputs: [Tensor<Element, Device>]) -> (Tensor<Element, Device>, Tensor<Element, Device>) {
        let encoderStateSequence = inputs[0] // [seqlen, batchSize, encHS]
        let decoderState = inputs[1] // [batchSize, decHS]
        
        // let seqlen = encoderStateSequence.shape[0]
        let batchSize = decoderState.shape[0]
        
        let encIn = encoderStateSequence.view(as: -1, encoderHiddenSize)
        
        let encScore = encIn.mmul(W_s).view(as: -1, batchSize, latentSize) // [seqlen * batchSize, latentSize]
        let decScore = decoderState.mmul(W_h) // [batchSize, latentSize]
        
        let scores = tanh(encScore + decScore + b) // [seqlen, batchSize, latentSize]
            .view(as: -1, latentSize) // [seqlen * batchSize, latentSize]
            .mmul(v.view(as: -1, 1)) // [seqlen * batchSize, 1]
            .view(as: -1, batchSize) // [seqlen, batchSize]
        
        if !isTemporal  {
            return (softmax(scores.T).T, scores)
        }
        
        let expScores = exp(scores) // [seqlen, batchSize]
        let attnHistory = inputs[2] // [seqlen, batchSize]
        
        if attnHistory.dim == 0 {
            return (softmax(scores.T).T, expScores)
        }
        
        return (normalize(expScores / attnHistory, along: 0), expScores + attnHistory)
    }
}

func normalize<Element, Device>(_ tensor: Tensor<Element, Device>, along axis: Int) -> Tensor<Element, Device> {
    return tensor / tensor.sum(axes: axis).unsqueeze(at: axis)
}

class AttentionCombine<Element: NumericType, Device: DeviceType>: Layer, Codable {
    var isTrainable: Bool {
        get {
            return false
        }
        set {
            // noop
        }
    }
    
    var parameters: [Tensor<Element, Device>] {
        return []
    }
    
    func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        let scores = inputs[0] // [seqlen, batchsize]
        let weights = inputs[1] // [seqlen, batchSize, hiddenSize]
        
        let scoredWeights = (scores.unsqueeze(at: 2) * weights).sum(axes: 0) // [batchSize, hiddenSize]
        
        return scoredWeights
    }
}

@dynamicCallable
class AttentionDecoder<Element: RandomizableType, Device: DeviceType>: Layer, Codable {
    typealias Input = Element
    
    var isTrainable: Bool = true
    
    var parameters: [Tensor<Element, Device>] {
        return Array([
            embed.parameters,
            attention.parameters,
            combine.parameters,
            rnn.parameters,
            deembed.parameters,
            softmax.parameters
        ].joined())
    }
    
    var trainableParameters: [Tensor<Element, Device>] {
        guard isTrainable else {
            return []
        }
        
        return Array([
            embed.trainableParameters,
            attention.trainableParameters,
            combine.trainableParameters,
            rnn.trainableParameters,
            deembed.trainableParameters,
            softmax.trainableParameters
        ].joined())
    }
    
    let embed: Embedding<Element, Device>
    let attention: TanhAttention<Element, Device>
    let combine: AttentionCombine<Element, Device>
    let rnn: GRU<Element, Device>
    let deembed: Dense<Element, Device>
    let softmax: Softmax<Element, Device>
    
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
        deembed = Dense(inputFeatures: hiddenSize, outputFeatures: inputSize)
        softmax = Softmax()
    }
    
    func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        fatalError("forward(_:) is unavailable, use forwardStep(input:state:encoderStates:) instead")
    }
    
    func forwardStep(input: Tensor<Int32, Device>, state: Tensor<Element, Device>, encoderStates: Tensor<Element, Device>, attentionHistory: Tensor<Element, Device>) -> (output: Tensor<Element, Device>, state: Tensor<Element, Device>, attentionScores: Tensor<Element, Device>, attentionHistory: Tensor<Element, Device>) {
        // input: [(1, )?batchSize] (squeezed at a later stage, so dim not important)
        // state: [batchSize, hiddenSize]
        // encoderState: [seqlen, batchSize, hiddenSize]
        
        let embedded = embed(input.view(as: -1)) // [batchSize, hiddenSize]
        
        let (attentionScores, history) = attention.forward([encoderStates, state, attentionHistory]) // [seqlen, batchSize]
        let weightedStates = combine(attentionScores, encoderStates) // [batchSize, hiddenSize]
        
        let rnnInput = Tensor.stack(embedded, weightedStates, axis: 1) // [batchSize, 2 * hiddenSize]
        let newState = rnn.step(x: rnnInput, state: state)
        
        let deembedded = deembed(newState)
        let wordScores = softmax(deembedded)
        
        return (output: wordScores, state: newState, attentionScores: attentionScores, attentionHistory: history)
    }
}
