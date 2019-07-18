//
//  Language.swift
//  DL4S
//
//  Created by Palle Klewitz on 15.03.19.
//

import Foundation
import DL4S


struct Language {
    static let startOfSentence: Int = 0
    static let endOfSentence: Int = 1
    static let unknown: Int = 2
    
    var wordToIndex: [String: Int]
    var words: [String]
    
    init(fromExamples examples: [String]) {
        let words = examples
            .flatMap {
                $0.components(separatedBy: .whitespaces)
            }
            .filter {!$0.isEmpty}
        
        let frequencies: [String: Int] = words.reduce(into: [:], {$0[$1, default: 0] += 1})
        
        let uniqueWords = Set(words)
            .sorted(by: {frequencies[$0, default: 0] < frequencies[$1, default: 0]})
            .reversed()
        
        self.words = ["<s>", "</s>", "<unk>"] + uniqueWords
        self.wordToIndex = Dictionary(uniqueKeysWithValues: self.words.enumerated().map {($1, $0)})
    }
    
    init(words: [String]) {
        self.words = words
        self.wordToIndex = Dictionary(uniqueKeysWithValues: self.words.enumerated().map {($1, $0)})
    }
    
    init(contentsOf url: URL) throws {
        self.words = try String(data: Data(contentsOf: url), encoding: .utf8)!
            .split(whereSeparator: {$0.isNewline})
            .map(String.init)
        self.wordToIndex = Dictionary(uniqueKeysWithValues: self.words.enumerated().map {($1, $0)})
    }
    
    func limited(toWordCount wordCount: Int) -> Language {
        return Language(words: Array(words[..<wordCount]))
    }
    
    func write(to url: URL) throws {
        try words.joined(separator: "\n")
            .data(using: .utf8)!
            .write(to: url)
    }
    
    static func cleanup(_ string: String) -> String {
        return string
            .lowercased()
            .replacingOccurrences(of: #"([.,?!();\-_$°+/:])"#, with: " $1 ", options: .regularExpression)
            .replacingOccurrences(of: #"[„"”“‟‘»«]"#, with: " \" ", options: .regularExpression)
            .replacingOccurrences(of: #"[–—－]"#, with: " - ", options: .regularExpression)
            .replacingOccurrences(of: #"[0-9]+"#, with: "<num>", options: .regularExpression)
    }
    
    static func pair(from path: String, maxLength: Int? = nil, replacements: ([String: String], [String: String])) throws -> (Language, Language, [(String, String)]) {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let string = String(data: data, encoding: .utf8)!
        
        let cleaned = Language.cleanup(string)
        
        let lines = cleaned.components(separatedBy: "\n")
            .filter {!$0.isEmpty}
        
        let pairs = lines
            .map {$0.components(separatedBy: "\t")}
            .map {($0[0], $0[1])}
            .filter { pair -> Bool in
                if let maxLength = maxLength {
                    return pair.0.components(separatedBy: .whitespaces).count <= maxLength
                } else {
                    return true
                }
            }
            .map {
                (
                    replacements.0.reduce($0, {$0.replacingOccurrences(of: $1.key, with: $1.value)}),
                    replacements.1.reduce($1, {$0.replacingOccurrences(of: $1.key, with: $1.value)})
                )
            }
        
        let l1 = Language(fromExamples: pairs.map {$0.0})
        let l2 = Language(fromExamples: pairs.map {$1})
        
        return (l1, l2, pairs)
    }
    
    func formattedSentence(from sequence: [Int32]) -> String {
        let words = sequence.filter {$0 >= 2}.map(Int.init).compactMap {self.words[safe: $0]}
        
        var result: String = ""
        
        for w in words {
            if let f = w.first, Set(".,!?").contains(f) {
                result.append(w)
            } else if result.isEmpty {
                result.append(w)
            } else {
                result.append(" \(w)")
            }
        }
        
        return result
    }
    
    func indexSequence(from sentence: String) -> [Int32] {
        let words = sentence.components(separatedBy: .whitespaces).filter {!$0.isEmpty}
        let indices = words.map {wordToIndex[$0] ?? Language.unknown}.map(Int32.init)
        return [Int32(Language.startOfSentence)] + indices + [Int32(Language.endOfSentence)]
    }
    
    func wordSequence(from indexSequence: [Int32]) -> [String] {
        return indexSequence.map(Int.init).compactMap {words[safe: $0]}
    }
}
