# Call-Me-Maybe

Introduction to function calling in LLMs

## All the info I gathered(Randomly typed)

A GPT or Generative Pre-trained Transformer is a bot that generates text based on a spesific mathematical calcualtion to guess the next word in a sentence, its called pre-trained because it has a lot of setted rules/parameters that can be fine tuned to better its performance.

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

#### Logits & Softmax

##### Logits

`Logits` or `logistic unit = log(odds)` is a calculated odd (in raw mode aka before softmax) for a token to be the next in a sentence.

Odds can be either a chance of how likely something will or will not happen;

Odds = Number of favorable outcomes / Number of unfavorable outcomes

Alternatively, if you know the probability of an event occurring, you can convert it to odds using these formulas:

`Odds in favor`: ``` P/1−P ```

`Odds against`: ``` 1−P/P ```

If odds are expressed as a ratio A:B (where A is chances for success and B is chances against), the probability of winning is calculated as:

``` P(Win)= A/A+B ```

##### Softmax

Softmax is a function that turns logits into probability distubution over the vocabulary.

This ensures all probabilities are between 0 and 1, and they sum to 1 for each position.

It's done by calculating it using this formula:

`probability(word) = e^(logit_for_word) / sum_of_all(e^(logit_for_each_word))`

##### Why not use probabilities directly instead of logits?

Well, its because most models use a cross-entropy function expects raw logits as input, not probabilities. Cross‑entropy compares the model’s raw scores with the correct token and pushes the model to increase the correct token’s logit and decrease others, wich is it's way of training the model.

#### Sampling Methods

After the logits are calculated and went through softmax, the sampling methods come next and uses one of the 3 algorithms to test the next word/token's compatibility with the others and how well it fits, it does that with one of these algorithms:

1- Top-k; Top‑k – sample randomly from the top k highest probabilities.
2- Top‑p; sample from the smallest set of tokens whose cumulative probability reaches p.
3- Greedy; pick the highest probability.

### The Generation loop

1- Prompt – User provides a string, e.g., "The capital of France is".

2- Encode – Tokenize the prompt into token IDs: [101, 345, 1200, ...].

3- Forward pass – Pass the token IDs through the model to obtain logits for the next token (at the last position).

4- Sample token – Convert logits to probabilities (softmax) and choose the next token according to some strategy (greedy, top‑k, top‑p, temperature).

5- Append – Add the new token ID to the end of the sequence.

6- Repeat – Feed the extended sequence back into the model, or more efficiently, use the cached key‑value states (KV cache) to only compute the new token.

Stop when:

An end‑of‑sequence token (`<eos>`) is generated.

Maximum length is reached.

A stop condition is met.

### Constraint decoding

constaint decoding is where we change the logits of words/tokens that we dont want in out output and set it's logit to -inf and that makes it equal to 0 after softmax and we keep the others that we want the same or in normal state, in some cases we increase. This forces the model to only sample the valid tokens.

### Finite-State Machine (FSM)
