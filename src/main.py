import torch
import os
import time
from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()
tokens = model.encode("The capital of France is")[0].tolist()
print(tokens)

temperature = 0.8
top_k = 50

for _ in range(100):
    logits = torch.tensor(model.get_logits_from_input_ids(tokens))

    # Top-k filtering
    values, indices = torch.topk(logits, top_k)

    # Apply temperature + softmax
    probs = torch.softmax(values / temperature, dim=0)

    # Sample instead of argmax
    next_token = indices[torch.multinomial(probs, 1)].item()

    tokens.append(next_token)

    decoded = model.decode(tokens)

    os.system("clear")
    print(decoded)
    # Stop if EOS token is generated
    if next_token == model._tokenizer.eos_token_id:
        break
