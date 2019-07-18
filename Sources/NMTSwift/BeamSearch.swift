//
//  BeamSearch.swift
//  NMTSwift
//
//  Created by Palle Klewitz on 03.05.19.
//

import Foundation
import DL4S


protocol StateType: Equatable {
    associatedtype Element
    
    func appending(_ element: Element) -> Self
}


struct Hypothesis<State: StateType> {
    var state: State
    var logProbability: Double
    var isCompleted: Bool
    var beamLength: Int
    
    init(initialState: State) {
        self.state = initialState
        self.logProbability = 0 // log(1)
        self.isCompleted = false
        self.beamLength = 0
    }
    
    init(state: State, logProbability: Double, isCompleted: Bool, beamLength: Int) {
        self.state = state
        self.logProbability = logProbability
        self.isCompleted = isCompleted
        self.beamLength = beamLength
    }
    
    func extended(with element: State.Element, logProbability: Double, isCompleted: Bool) -> Hypothesis<State> {
        guard !self.isCompleted else {
            return self
        }
        return Hypothesis(state: self.state.appending(element), logProbability: self.logProbability + logProbability, isCompleted: isCompleted, beamLength: self.beamLength + 1)
    }
}

struct BeamSearchContext<State: StateType> {
    private(set) var hypotheses: [Hypothesis<State>]
    private var newHypotheses: [Hypothesis<State>] = []
    
    let maxBeamLength: Int
    let beamCount: Int
    
    /// Hypotheses, sorted by likelihood descending
    var bestHypotheses: [Hypothesis<State>] {
        return hypotheses
            .sorted(by: {$0.logProbability < $1.logProbability})
            .reversed()
    }
    
    var isCompleted: Bool {
        return hypotheses.allSatisfy {
            $0.isCompleted || $0.beamLength >= maxBeamLength
        }
    }
    
    init(beamCount: Int, maxLength: Int, initialState: State) {
        self.beamCount = beamCount
        self.maxBeamLength = maxLength
        self.hypotheses = [Hypothesis(initialState: initialState)]
    }
    
    mutating func endIteration() {
        hypotheses = newHypotheses
            .sorted(by: {$0.logProbability < $1.logProbability})
            .reversed()
            .prefix(beamCount)
            .collect(Array.init)
        newHypotheses = []
    }
    
    mutating func add(_ hypothesis: Hypothesis<State>) {
        newHypotheses.append(hypothesis)
    }
}
