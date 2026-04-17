# Call-Me-Maybe

Introduction to function calling in LLMs

## All the info I gathered

A GPT or Generative Pre-trained Transformer is a bot that generates text based on a spesific mathematical calcualtion to guess the next word in a sentence, its called pre-trained bc it has a lot of setted rules/parameters that can be fine tuned to better its performance

A LLm or Large Language Model is a type of neural network that was trained on enormous size of text based data that could be found online.

what is a neural network?
It's a series of algorithems that try to recognise patterns in data, something similar to what your brain do; how you can recognize an image or the idea behind an image even if the images are diffrent.

So LLM basically is a type of nueral network that it main goel is to recognise patterns in text based data and the way of doing that is by surfing the internet and learning instaed of you giving it a set .

The way int works is by following a set of steps:

### 1-Tokenization

#### What is tokenization?

It's a proccess that take a piece of text and break it down into a cequence of integers called token IDs that a model can understand. Since models operate on numbers, not characters or words, we need a mapping from text fragments to IDs.

The model at hand along with most modern models nowadays use subword tokenization and its a way to break a long word into adjacent pairs that are frenquently seen with eachother (Greedy pair merging) ex: 'e' & 'st' are often seen in words toghether ('est') so if a word like "longest" is in process it get split into 'long' & 'est'. This can be achieved using an algorithm called `BPE (Byte-Pair Encoding)` and it does this untill a predefined vocabulary size is reached.

After many merges, you get a compact vocabulary that can represent unseen words by splitting them into known subwords.

There is also anothor algorithm called `SentencePiece` and it basiclly does the same job as BPE, the diffrence between them is that BPE requires a pre-tokenized text(words split by spaces/punctuation), but SetencePiece works directly on RAW text.

Also BPE use the Greedy pair merging algo but SentencePiece can either use BPE's way or unigram language model segmentation's.

#### How words become token IDs?

`1-`The text is normalized (lowercase, Unicode normalization, etc.).

`2-`It is split into subwords using the learned merge rules (BPE) or a unigram model (SentencePiece).

`3-`Each subword is looked up in a vocabulary – a dictionary mapping subword strings to unique integers.

`4-`The resulting list of integers is the input_ids fed to the model.

```bash
Example (BPE):
Input: "unhappiness"
Tokens: ["un", "happiness"] or ["un", "happy", "ness"] depending on merges.
IDs: [347, 8921]
```

`Logits` or `logistic unit = log(odds)` is a calculated odd (in raw mod aka before softmax) for a token to be the next in a sentence.
