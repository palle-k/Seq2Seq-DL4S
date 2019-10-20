# Seq2Seq with Attention using DL4S

An implementation of seq2seq for neural machine translation in Swift based on DL4S.

Seq2seq uses two recurrent neural networks, an encoder and a decoder, which are trained in an end to end setup:
The encoder creates a latent vector representation of the input sequence, the decoder then produces an output sequence based on this latent representation.

Both networks are updated together by backpropagating the decoding loss through the combined model.

The input sequence is a sequence of word indices corresponding to words in the source language.
The output sequence is a sequence of probability distributions of words in the target language.

### Attention

In a vanilla seq2seq model, the final hidden state of the decoder has to carry all the information contained in the input sequence, which limits
the capacity of the model. To overcome this, the decoder can recall past encoder states.

This implementation provides tanh and general attention modules (following Luong et al.). 
Both attention modes optionally allow temporal attention,
where the attention module is forced to focus its attention onto different states.

## Source Code Overview

- The encoder and decoder are defined in Sources/NMTSwift/Models.swift and both use gated recurrent units, which are computationally more efficent compared to LSTMs.
- The encoding and decoding process is coordinated by the Helper type defined at Sources/NMTSwift/Helpers.swift

## Usage

For a detailed usage description and additional options run 

```bash
swift run NMTSwift --help
swift run NMTSwift [subcommand] --help
```

### Input

Download the desired file of sentence pairs from [ManyThings.org](https://www.manythings.org/anki/).
Alternatively, any file of examples can be used as long as it follows the supported input format.

```
Example in source language\tExample in destination language
```

### Train a model

It is recommended to run everything in release configuration (`-c release`)

```bash
swift run -c release NMTSwift train eng-ger.txt ./model.json  --logdir ./logs
```

### Translation Web App

A trained model can be tested using the translation web app, which is included in this repository.
To build the translation app, navigate into the `static` folder and run `npm run build`.

Then navigate back to the repository root and run the server using the following command:

```bash
swift run -c release NMTSwift eval eng-ger.txt models/encoder.json models/decoder.json
```

### Evaluate  model

Evaluation requires Python 3, numpy and matplotlib to be installed to visualize the attention distribution.

```bash
swift run -c release NMTSwift eval eng-ger.txt models/encoder.json models/decoder.json
```

Running this command will start a interactive translation session.
