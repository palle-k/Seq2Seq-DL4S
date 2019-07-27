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
        return Seq2SeqAttentionBeamState(indices: indices + [element.word], hiddenState: element.hiddenState, attentions: attentions + [element.attention], attentionSum: attentionSum)
    }
}


class AttentionSeq2Seq<Element: RandomizableType, Device: DeviceType>: Layer {
    typealias Input = Int32
    
    var isTrainable: Bool = true
    
    let encoder: BidirectionalEncoder<Element, Device>
    let decoder: AttentionDecoder<Element, Device>
    let transformHiddenState: Sequential<Element, Device>
    
    var parameters: [Tensor<Element, Device>] {
        return Array([encoder.parameters, decoder.parameters, transformHiddenState.parameters].joined())
    }
    
    var trainableParameters: [Tensor<Element, Device>] {
        guard isTrainable else {
            return []
        }
        return Array([encoder.trainableParameters, decoder.trainableParameters, transformHiddenState.trainableParameters].joined())
    }
    
    init(encoder: BidirectionalEncoder<Element, Device>, decoder: AttentionDecoder<Element, Device>, transformHiddenState: Sequential<Element, Device>) {
        self.encoder = encoder
        self.decoder = decoder
        self.transformHiddenState = transformHiddenState
    }
    
    func decodeSequence(fromInitialState initialState: Tensor<Element, Device>, encoderStates: Tensor<Element, Device>, initialToken: Int32, endToken: Int32, maxLength: Int) -> (decoded: Tensor<Element, Device>, attentionScores: Tensor<Element, Device>) {
        var hiddenState = initialState
        var token = initialToken
        var sequence: [Tensor<Element, Device>] = []
        var attnSum = Tensor<Element, Device>(0)
        var attentionScores: [Tensor<Element, Device>] = []
        for _ in 0 ..< maxLength {
            let (out, h, attn, asum) = decoder.forwardStep(input: Tensor([token]), state: hiddenState, encoderStates: encoderStates, attentionHistory: attnSum)
            sequence.append(out) // out has a shape of [1, vocabularySize]
            hiddenState = h
            attnSum = asum
            token = Int32(argmax(out))
            attentionScores.append(attn.squeeze().unsqueeze(at: 0))
            
            if token == endToken {
                break
            }
        }
        
        return (stack(sequence), stack(attentionScores))
    }
    
    func decodeSequence(fromInitialState initialState: Tensor<Element, Device>, encoderStates: Tensor<Element, Device>, forcedResult: [Int32]) -> (decoded: Tensor<Element, Device>, attentionScores: Tensor<Element, Device>) {
        var hiddenState = initialState
        var sequence: [Tensor<Element, Device>] = []
        var attentionScores: [Tensor<Element, Device>] = []
        var attnSum = Tensor<Element, Device>(0)
        
        for token in forcedResult {
            let (out, h, attn, asum) = decoder.forwardStep(input: Tensor([token]), state: hiddenState, encoderStates: encoderStates, attentionHistory: attnSum)
            hiddenState = h
            attnSum = asum
            sequence.append(out)
            attentionScores.append(attn.squeeze().unsqueeze(at: 0))
        }
        
        return (stack(sequence), stack(attentionScores))
    }
    
    func beamDecode(fromInitialState initialState: Tensor<Element, Device>, encoderStates: Tensor<Element, Device>, initialToken: Int32, endToken: Int32, maxLength: Int, beamCount: Int) -> [Seq2SeqAttentionBeamState<Element, Device>] {
        var context = BeamSearchContext(
            beamCount: beamCount,
            maxLength: maxLength,
            initialState: Seq2SeqAttentionBeamState(indices: [initialToken], hiddenState: initialState, attentions: [], attentionSum: Tensor(0))
        )
        
        while !context.isCompleted {
            for hypothesis in context.hypotheses {
                if hypothesis.isCompleted {
                    context.add(hypothesis)
                    continue
                }
                
                let (out, h, attn, attnSum) = decoder.forwardStep(
                    input: Tensor([hypothesis.state.indices.last!]),
                    state: hypothesis.state.hiddenState,
                    encoderStates: encoderStates,
                    attentionHistory: hypothesis.state.attentionSum
                )
                
                let best = out
                    .flattenedArray
                    .enumerated()
                    .max(count: beamCount, by: {$0.element < $1.element})
                
                for (idx, prob) in best {
                    context.add(
                        hypothesis.extended(
                            with: (
                                word: Int32(idx),
                                hiddenState: h,
                                attention: attn.squeeze().unsqueeze(at: 0),
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
    
    func translate(_ text: [Int32], from sourceLanguage: Language, to destinationLanguage: Language, beamCount: Int) -> [([Int32], Tensor<Element, Device>)] {
//        let encoded = encoder(Tensor<Input, Device>(text))
//        let initialState = transformHiddenState(encoded[-1])
//
//        let states = self.beamDecode(
//            fromInitialState: initialState,
//            encoderStates: encoded,
//            initialToken: Int32(Language.startOfSentence),
//            endToken: Int32(Language.endOfSentence),
//            maxLength: text.count * 2,
//            beamCount: beamCount
//        )
//        return states.map {state in (state.indices, stack(state.attentions))}
        
        let (decoded, attn) = forwardWithScores([
            Tensor<Input, Device>(text)
        ])
        
        return [(Helper<Element, Device>().sequence(from: decoded), attn)]
    }
    
    func forwardWithScores(_ inputs: [Tensor<Int32, Device>]) -> (decoded: Tensor<Element, Device>, attentionScores: Tensor<Element, Device>) {
        let encoded = encoder(inputs[0])
        let initialState = transformHiddenState(encoded[-1])
        
        if inputs.count == 2 {
            return decodeSequence(fromInitialState: initialState, encoderStates: encoded, forcedResult: inputs[1].flattenedArray)
        } else {
            return decodeSequence(fromInitialState: initialState, encoderStates: encoded, initialToken: Int32(Language.startOfSentence), endToken: Int32(Language.endOfSentence), maxLength: inputs[0].shape[0] * 2)
        }
    }
    
    func forward(_ inputs: [Tensor<Int32, Device>]) -> Tensor<Element, Device> {
        return forwardWithScores(inputs).decoded
    }
}
