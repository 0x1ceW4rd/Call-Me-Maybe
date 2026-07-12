*This project has been created as part of the 42 curriculum by aezzirar.*

# Procedural Constrained Decoding for Function Calling

## Description

This project forces a small 0.6B‑parameter language model (Qwen3‑0.6B) to output **100% valid, schema‑compliant JSON function calls**.  
Instead of relying on prompt engineering, it intercepts the model’s raw logits and applies mathematical masks that **forbid invalid tokens** at every generation step. This guarantees that the output is always parseable and matches the required function signature.

## How It Works (Simplified)

1. **We inject the JSON structure** – the keys like `"name": "` and `"parameters": {` are written directly into the prompt. The LLM only generates the actual values (function name, arguments).
2. **For each value**, we use a pre‑computed mask to allow only certain tokens:
   - For **function names** – only tokens that appear in the given function list.
   - For **numbers** – only digits, minus sign, dot (but never a comma inside the number).
   - For **strings** – any character except an unescaped double quote, which acts as a stop signal.
3. **The generation loop** checks each token; when a stop character (like `"` or `,`) appears, the loop exits and we inject the next JSON key.
4. **No invalid JSON** can ever be produced because the model is never allowed to emit a token that would break the schema.

## Installation & Execution

### Prerequisites

- Python 3.10+
- `uv` (package manager)

### Makefile rules

```bash
make setup     # Setups the venv on the goingfre
make install   # install dependencies (uv sync)
make run       # run with default input files
make debug     # run with pdb
make clean     # remove cache files
make lint      # flake8 + mypy (basic)
make lint-strict # flake8 + mypy --strict
```

### Manual run

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json \
    --model Qwen/Qwen3-0.6B
```

All arguments are optional; default paths are used if omitted.

### Key Design Choices

Pre‑computed NumPy masks – built once at startup (O(1) per step) to avoid slow token loops.

“Comma hack” – blocks tokens that contain both digits and a comma, so the model outputs numbers and commas separately; this makes it easy to stop generation when a comma appears.

Dynamic few‑shot prompting – injects task‑specific examples into the prompt to guide the model’s natural probabilities toward the correct function.

Strict Pydantic validation – input files are validated against schemas; invalid files are reported cleanly.

### Performance & Reliability

100% valid JSON – guaranteed by construction.

Speed – processes ~10 prompts in under 5 seconds on a standard CPU (no GPU).

Memory – < 2 GB RAM (inference‑only, no gradients).

Supported types – string, integer, number.
(Boolean is not implemented; the code will ignore it if present.)

### Challenges & Solutions

Tokenizer quirks – Qwen adds special tokens like Ġ (space) and Ċ (newline). We strip these during mask building.

Escape handling – strings may contain \"; our generation loop correctly distinguishes escaped quotes from the closing quote.

Number extraction – we collect tokens until a comma appears, then validate the whole string before converting to float or int. Invalid numbers cause a clean error exit.

## Resources

[Qwen3‑0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) – model and tokenizer.

[Pydantic](https://pydantic.dev/docs/) – schema validation.

[NumPy](https://numpy.org/) – array operations for masks.

### AI Usage

Used ChatGPT to understand constrained decoding and logit masking.

Copilot assisted with boilerplate (argparse, file I/O). All generated code was reviewed, understood, and modified where necessary.

AI helped debug tokenizer special characters and escape handling.

Peer review was done to catch blind spots.
