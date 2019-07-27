//
//  main.swift
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

import DL4S
import Foundation
import Commander

let engReplacements = [
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "isn't": "is not",
    "aren't": "are not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "can't": "cannot",
    "don't": "do not",
    "didn't": "did not",
    "won't": "will not",
    "wasn't": "was not",
    "weren't": "were not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "mustn't": "must not"
]

let gerReplacements = [
    "kann's": "kann es",
    "wird's": "wird es",
    "bring's": "bring es",
    "werd's": "werde es",
    "mach's": "mach es",
    "macht's": "macht es",
    "interessiert's": "interessiert es",
    "ich's": "ich es",
    "schmeckt's": "schmeckt es",
    "ging's": "ging es",
    "hab's": "habe es",
    "läuft's": "läuft es",
    "tu's": "tu es",
    "halt's": "halte es",
    "kontrolliert's": "kontrolliert es",
    "mag's": "mag es",
    "sich's": "sich es",
    "hol's": "hol es"
]


let group = Group { g in
    g.command(
        "train",
        // Required args
        Argument<String>("dataset", description: "Path to tab separated dataset file", validator: validatePathExists(isDirectory: false)),
        Argument<String>("destination", description: "Directory to write the final trained model file to"),
        // Embedding parameters
        Option<Int?>("vocab_limit", default: nil, description: "Maximum number of words in vocabularies", validator: validatePositive(message: "Vocabulary size must be positive")),
        Option<String?>("src_emb", default: nil, description: "Location of word embeddings in the source language", validator: validatePathExists(isDirectory: false)),
        Option<String?>("dst_emb", default: nil, description: "Location of word embeddings in the destination language", validator: validatePathExists(isDirectory: false)),
        Option<Int>("emb_size", default: 100, description: "Size of embedded words", validator: validatePositive(message: "Embedding dimensionality must be greater than 0.")),
        // Model parameters
        Option<Float>("lr", default: 0.001, flag: nil, description: "Learning rate", validator: validatePositive(message: "Learning rate must be positive")),
        Option<Int>("latent_size", default: 1024, flag: nil, description: "Size of latent sentence representation", validator: validatePositive(message: "Latent size must be positive")),
        Option<Float>("forcing_rate", default: 0.5, flag: nil, description: "Teacher forcing rate", validator: validateRange(0 ... 1, message: "Teacher forcing rate must be between 0 and 1")),
        // Training parameters
        Option<Int>("iterations", default: 10000, flag: "i", description: "Number of training iterations", validator: validatePositive(message: "Number of training iterations must be positive")),
        Option<Int>("batch_size", default: 50, description: "Number of samples in a batch", validator: validatePositive(message: "Batch size must be positive")),
        Option<String>("checkpoint_dir", default: "./", flag: nil, description: "Directory to write checkpoints to", validator: validatePathExists(isDirectory: true)),
        Option<Int>("checkpoint_frequency", default: 2500, flag: nil, description: "Number of iterations between checkpoints (set to zero if no checkpoints should be created)", validator: validatePositive(message: "Number of iterations between checkpoints must be positive (set to zero if no checkpoints should be created)")),
        Option<String>("log_dir", default: "./", flag: nil, description: "Directory to write logs to", validator: validatePathExists(isDirectory: true)),
        description: "Trains a Seq2Seq model"
    ) { datasetPath, destinationPath, vocabLimit, srcEmbeddings, dstEmbeddings, embSize, learningRate, latentSize, teacherForcingRate, iterations, batchSize, checkpointDir, checkpointFrequency, logDir in
        
        let checkpointDir = URL(fileURLWithPath: checkpointDir)
        
        print("Loading dataset...", terminator: "")
        fflush(stdout)
        let (english, german, examples) = try { () -> (Language, Language, [(String, String)]) in
            var (eng, ger, ex) = try Language.pair(
                from: datasetPath,
                replacements: (engReplacements, gerReplacements)
            )
            if let vocabLimit = vocabLimit {
                eng = eng.limited(toWordCount: vocabLimit)
                ger = ger.limited(toWordCount: vocabLimit)
            }
            return (eng, ger, ex)
        }()
        
        try english.write(to: URL(fileURLWithPath: "vocab_eng.txt", relativeTo: checkpointDir))
        try german.write(to: URL(fileURLWithPath: "vocab_ger.txt", relativeTo: checkpointDir))
        print(" Done.")
        
        print("Creating model...", terminator: "")
        fflush(stdout)
        
        let encoder = BidirectionalEncoder<Float, CPU>(
            inputSize: english.words.count,
            embeddingSize: embSize,
            hiddenSize: latentSize,
            embeddingsFile: srcEmbeddings.map(URL.init(fileURLWithPath:)),
            words: english.words
        )
        let decoder = AttentionDecoder<Float, CPU>(
            inputSize: german.words.count,
            embeddingSize: embSize,
            hiddenSize: latentSize,
            words: german.words,
            embeddingsFile: dstEmbeddings.map(URL.init(fileURLWithPath:)),
            useTemporalAttention: true
        )
        let mapper = Sequential<Float, CPU>(
            Dense(inputFeatures: latentSize, outputFeatures: latentSize).asAny(),
            Relu().asAny()
        )
        let combined = AttentionSeq2Seq(encoder: encoder, decoder: decoder, transformHiddenState: mapper)
        print(" Done.")
        
        let optim = Adam(parameters: combined.trainableParameters, learningRate: learningRate)
        
        let epochs = iterations
        
        var progressBar = DL4S.ProgressBar<String>(totalUnitCount: epochs, formatUserInfo: {$0}, label: "training")
        let writer = try SummaryWriter(destination: URL(fileURLWithPath: logDir), runName: ISO8601DateFormatter().string(from: Date()))
        // let writer = TBSummaryWriter(runName: "runs/\(ISO8601DateFormatter().string(from: Date()))")
        
        for i in 1 ... epochs {
            optim.zeroGradient()
            
            var combinedLoss = Tensor<Float, CPU>(0)
            
            var src: String = ""
            var seq: String = ""
            
            for _ in 0 ..< batchSize {
                let (eng, ger) = examples.randomElement()!
                
                let engIdxs = english.indexSequence(from: eng)
                let gerIdxs = german.indexSequence(from: ger)
                
                let decoded: Tensor<Float, CPU>
                
                let teacherForcing = Float.random(in: 0 ... 1) <= teacherForcingRate
                
                if teacherForcing {
                    decoded = combined.forward(Tensor<Int32, CPU>(engIdxs), Tensor<Int32, CPU>(gerIdxs))
                } else {
                    decoded = combined.forward(Tensor<Int32, CPU>(engIdxs))
                }
                
                combinedLoss += Helper<Float, CPU>().decodingLoss(forExpectedSequence: Array(gerIdxs.dropFirst()), actualSequence: decoded)
                
                src = english.formattedSentence(from: engIdxs)
                seq = german.formattedSentence(from: Helper<Float, CPU>().sequence(from: decoded))
            }
            
            combinedLoss = combinedLoss / Tensor(Float(batchSize))
            combinedLoss.backwards()
            optim.step()
            
            // Prevents stack overflow when releasing compute graph lol
            combinedLoss.detachAll()
            
            progressBar.next(userInfo: "[loss: \(combinedLoss)] (\(src) -> \(seq))")
            writer.write(combinedLoss.item, named: "loss", at: i)
            // writer.addScalar(name: "loss", value: Double(combinedLoss.item), iteration: i)
            
            if checkpointFrequency > 0 && i.isMultiple(of: checkpointFrequency) && i != epochs {
                try combined.saveWeights(to: URL(fileURLWithPath: "model_\(i).json", relativeTo: checkpointDir))
            }
        }
        progressBar.complete()
        
        let destinationURL = URL(fileURLWithPath: destinationPath)
        try combined.saveWeights(to: destinationURL)
    }
    
    g.command(
        "eval",
        //Argument<String>("dataset", description: "Path to dataset", validator: validatePathExists(isDirectory: false)),
        Argument<String>("src_vocab", description: "Path to source vocabulary", validator: validatePathExists(isDirectory: false)),
        Argument<String>("dst_vocab", description: "Path to destination vocabulary", validator: validatePathExists(isDirectory: false)),
        Argument<String>("model_file", description: "Path to the stored weights of the model", validator: validatePathExists(isDirectory: false)),
        Option<Int>("emb_size", default: 100, description: "Size of embedded words", validator: validatePositive(message: "Embedding dimensionality must be greater than 0.")),
        Option<Int>("latent_size", default: 1024, flag: nil, description: "Size of latent sentence representation", validator: validatePositive(message: "Latent size must be positive")),
        Option<Int>("beam_count", default: 4, flag: nil, description: "Number of beams to use for decoding", validator: validatePositive(message: "Number of beams must be positive")),
        description: "Evaluate a trained seq2seq model."
    ) { srcVocab, dstVocab, modelPath, embSize, latentSize, beamCount in
        print("Loading vocabulary...", terminator: "")
        fflush(stdout)
        let english = try Language(contentsOf: URL(fileURLWithPath: srcVocab))
        let german = try Language(contentsOf: URL(fileURLWithPath: dstVocab))
//        let (english, german, pairs) = try Language.pair(from: dsPath, replacements: (engReplacements, gerReplacements))
        print(" Done.")
        
        print("Creating model...", terminator: "")
        fflush(stdout)
        let encoder = BidirectionalEncoder<Float, CPU>(
            inputSize: english.words.count,
            embeddingSize: embSize,
            hiddenSize: latentSize,
            embeddingsFile: nil,
            words: english.words
        )
        let decoder = AttentionDecoder<Float, CPU>(
            inputSize: german.words.count,
            embeddingSize: embSize,
            hiddenSize: latentSize,
            words: german.words,
            embeddingsFile: nil,
            useTemporalAttention: true
        )
        let mapper = Sequential<Float, CPU>(
            Dense(inputFeatures: latentSize, outputFeatures: latentSize).asAny(),
            Relu().asAny()
        )
        let combined = AttentionSeq2Seq(encoder: encoder, decoder: decoder, transformHiddenState: mapper)
        print(" Done.")
        
        print("Loading trained model...", terminator: "")
        fflush(stdout)
        try combined.loadWeights(from: URL(fileURLWithPath: modelPath))
        print(" Done.")
        
        print("> ", terminator: "")
        fflush(stdout)
        while let line = readLine() {
            let cleaned = engReplacements.reduce(line.lowercased(), {$0.replacingOccurrences(of: $1.key, with: $1.value)})

            let input = Language.cleanup(cleaned)

            let inputIdxs = english.indexSequence(from: input)

            let translations = combined.translate(inputIdxs, from: english, to: german, beamCount: beamCount)
            let (translated, attn) = translations[0]

            for (t, _) in translations {
                print(german.formattedSentence(from: t))
            }

            Plot.showAttention(attn, source: english.wordSequence(from: inputIdxs), destination: german.wordSequence(from: translated))

            print("> ", terminator: "")
            fflush(stdout)
        }
        
//        while true {
//            let (src, dst) = pairs.randomElement()!
//            let engIdxs = english.indexSequence(from: src)
//            let gerIdxs = german.indexSequence(from: dst)
//
//            let (decoded, attn) = combined.forwardWithScores([Tensor<Int32, CPU>(engIdxs), Tensor<Int32, CPU>(gerIdxs)])
//            let translated = Helper<Float, CPU>().sequence(from: decoded)
//
//            Plot.showAttention(attn, source: ["<s>"] + src.components(separatedBy: .whitespaces) + ["</s>"], destination: german.wordSequence(from: translated))
//
//            print("\(english.formattedSentence(from: engIdxs)) -> \(german.formattedSentence(from: translated))")
//        }
    }
}

group.run()

