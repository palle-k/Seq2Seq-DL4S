//
//  AttentionSeq2Seq.swift
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


struct Seq2SeqAttentionBeamState<Element: NumericType, Device: DeviceType>: StateType {
    typealias Element = (word: Int32, hiddenState: Tensor<Element, Device>, attention: Tensor<Element, Device>, attentionSum: Tensor<Element, Device>)
    
    var indices: [Int32]
    var hiddenState: Tensor<Element, Device>
    var attentions: [Tensor<Element, Device>]
    var attentionSum: Tensor<Element, Device>
    
    func appending(_ element: (word: Int32, hiddenState: Tensor<Element, Device>, attention: Tensor<Element, Device>, attentionSum: Tensor<Element, Device>)) -> Seq2SeqAttentionBeamState<Element, Device> {
        return Seq2SeqAttentionBeamState(indices: indices + [element.word], hiddenState: element.hiddenState, attentions: attentions + [element.attention], attentionSum: element.attentionSum)
    }
}


public struct AttentionSeq2Seq<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    public var parameters: [Tensor<Element, Device>] {
        Array([
            encoder.parameters,
            decoder.parameters
        ].joined())
    }
    
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {
        Array([
            encoder.parameterPaths.map((\Self.encoder).appending(path:)),
            decoder.parameterPaths.map((\Self.decoder).appending(path:))
        ].joined())
    }
    
    public var encoder: BidirectionalEncoder<Element, Device>
    public var decoder: AttentionDecoder<Element, Device>
    
    public init(encoder: BidirectionalEncoder<Element, Device>, decoder: AttentionDecoder<Element, Device>) {
        self.encoder = encoder
        self.decoder = decoder
    }
    
    func decodeSequence(encoderStates: Tensor<Element, Device>, forcedResult: [Int32]) -> (decoded: Tensor<Element, Device>, attentionScores: Tensor<Element, Device>) {
        var hiddenState = decoder.initialState(forBatchSize: 1)
        var sequence: [Tensor<Element, Device>] = []
        var attentionScores: [Tensor<Element, Device>] = []
        var attnSum = Tensor<Element, Device>(0)
        
        for token in forcedResult {
            let (out, h, attn, asum) = decoder.callAsFunction((input: Tensor([token]), state: hiddenState, encoderStates: encoderStates, attentionHistory: attnSum))
            hiddenState = h
            attnSum = asum
            sequence.append(out)
            attentionScores.append(attn.squeezed().unsqueezed(at: 0))
        }
        
        return (stack(sequence), stack(attentionScores))
    }
    
    func decodeSequence(encoderStates: Tensor<Element, Device>, initialToken: Int32, endToken: Int32, maxLength: Int) -> (decoded: Tensor<Element, Device>, attentionScores: Tensor<Element, Device>) {
        var hiddenState = decoder.initialState(forBatchSize: 1)
        var token = initialToken
        var sequence: [Tensor<Element, Device>] = []
        var attnSum = Tensor<Element, Device>(0)
        var attentionScores: [Tensor<Element, Device>] = []
        for _ in 0 ..< maxLength {
            let (out, h, attn, asum) = decoder.callAsFunction((input: Tensor([token]), state: hiddenState, encoderStates: encoderStates, attentionHistory: attnSum))
            sequence.append(out) // out has a shape of [1, vocabularySize]
            hiddenState = h
            attnSum = asum
            
            token = Int32(out.argmax())
            attentionScores.append(attn.squeezed().unsqueezed(at: 0))
            
            if token == endToken {
                break
            }
        }
        
        return (stack(sequence), stack(attentionScores))
    }
    
    func decodeSequence(encoderStates: Tensor<Element, Device>, initialToken: Int32, forcedResult: [Int32], teacherForcingRatio: Float) -> (decoded: Tensor<Element, Device>, attentionScores: Tensor<Element, Device>) {
        var hiddenState = decoder.initialState(forBatchSize: 1)
        var sequence: [Tensor<Element, Device>] = []
        var attentionScores: [Tensor<Element, Device>] = []
        var attnSum = Tensor<Element, Device>(0)
        
        var currentToken = initialToken
        
        for token in forcedResult {
            let input: Tensor<Int32, Device>
            if Float.random(in: 0 ... 1) <= teacherForcingRatio {
                input = Tensor([token])
            } else {
                input = Tensor([currentToken])
            }
            
            let (out, h, attn, asum) = decoder.callAsFunction((
                input: input,
                state: hiddenState,
                encoderStates: encoderStates,
                attentionHistory: attnSum
            ))
            hiddenState = h
            attnSum = asum
            sequence.append(out)
            attentionScores.append(attn.squeezed().unsqueezed(at: 0))
            
            currentToken = Int32(out.argmax())
        }
        
        return (stack(sequence), stack(attentionScores))
    }
    
    func beamDecode(encoderStates: Tensor<Element, Device>, initialToken: Int32, endToken: Int32, maxLength: Int, beamCount: Int) -> [Seq2SeqAttentionBeamState<Element, Device>] {
        var context = BeamSearchContext(
            beamCount: beamCount,
            maxLength: maxLength,
            initialState: Seq2SeqAttentionBeamState(
                indices: [initialToken],
                hiddenState: decoder.initialState(forBatchSize: 1),
                attentions: [], attentionSum: Tensor(0)
            )
        )
        
        while !context.isCompleted {
            for hypothesis in context.hypotheses {
                if hypothesis.isCompleted {
                    context.add(hypothesis)
                    continue
                }
                let (out, h, attn, attnSum) = decoder.callAsFunction((
                    input: Tensor([hypothesis.state.indices.last!]),
                    state: hypothesis.state.hiddenState,
                    encoderStates: encoderStates,
                    attentionHistory: hypothesis.state.attentionSum
                ))
                
                let best = out
                    .elements
                    .enumerated()
                    .top(count: beamCount, by: {$0.element < $1.element})
                
                for (idx, prob) in best {
                    context.add(
                        hypothesis.extended(
                            with: (
                                word: Int32(idx),
                                hiddenState: h,
                                attention: attn.squeezed().unsqueezed(at: 0),
                                attentionSum: attnSum
                            ),
                            logProbability: Double(element: prob.log()),
                            isCompleted: idx == Int(endToken)
                        )
                    )
                }
            }
            
            context.endIteration()
        }
        
        return context.bestHypotheses.map {$0.state}
    }
    
    public func translate(_ text: [Int32], from sourceLanguage: Language, to destinationLanguage: Language, beamCount: Int) -> [([Int32], Tensor<Element, Device>)] {
        let encoded = encoder.callAsFunction(Tensor<Int32, Device>(text))
        let states = self.beamDecode(
            encoderStates: encoded,
            initialToken: Int32(Language.startOfSentence),
            endToken: Int32(Language.endOfSentence),
            maxLength: text.count * 2,
            beamCount: beamCount
        )
        return states.map {state in (state.indices.dropFirst().collect(Array.init), stack(state.attentions))}
    }
    
    public func callAsFunction(_ inputs: (input: Tensor<Int32, Device>, target: Tensor<Int32, Device>, teacherForcingRatio: Float)) -> (decoded: Tensor<Element, Device>, attentionScores: Tensor<Element, Device>) {
        let encoded = encoder(inputs.input)
        
        return decodeSequence(encoderStates: encoded, initialToken: Int32(Language.startOfSentence), forcedResult: inputs.target.elements, teacherForcingRatio: inputs.teacherForcingRatio)
    }
    
}
