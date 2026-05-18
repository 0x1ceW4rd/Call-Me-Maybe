import numpy as np
from llm_sdk import Small_LLM_Model

class ConstrainedGenerator:
    def __init__(self, model: Small_LLM_Model, function_names: list[str], vocab: dict[str, int]):
        self.model = model
        self.function_names = function_names   # not used, kept for compatibility

    def _to_flat_int_list(self, ids):
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(x) for x in ids]

    def generate(self, prompt: str, max_new_tokens: int = 150, temperature: float = 0.8, top_k: int = 50) -> str:
        raw_ids = self.model.encode(prompt)
        tokens = self._to_flat_int_list(raw_ids)
        generated = ""

        for _ in range(max_new_tokens):
            logits_tensor = self.model.get_logits_from_input_ids(tokens)
            if hasattr(logits_tensor, 'detach'):
                logits = logits_tensor.detach().cpu().numpy()
            else:
                logits = np.array(logits_tensor)
            if logits.ndim == 3:
                logits = logits[0, -1, :]
            elif logits.ndim == 2:
                logits = logits[-1, :]
            else:
                logits = logits.flatten()

            if top_k < len(logits):
                topk_indices = np.argpartition(logits, -top_k)[-top_k:]
                topk_vals = logits[topk_indices]
            else:
                topk_indices = np.arange(len(logits))
                topk_vals = logits

            probs = np.exp(topk_vals / temperature)
            probs /= probs.sum()
            chosen_idx = np.random.choice(len(topk_indices), p=probs)
            next_token_id = int(topk_indices[chosen_idx])

            eos_id = self.model._tokenizer.eos_token_id
            if next_token_id == eos_id:
                break

            tokens.append(next_token_id)
            new_text = self.model.decode([next_token_id])
            generated += new_text

            # Stop when we have a complete JSON object (balanced braces)
            if generated.count('{') > 0 and generated.count('{') == generated.count('}'):
                break

        return generated