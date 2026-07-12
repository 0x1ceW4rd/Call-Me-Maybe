import numpy as np
from llm_sdk import Small_LLM_Model
from src.models import Functions
from typing import Dict, List, Set, Any
import numpy.typing as npt


class VocabManager:
    def __init__(self, model: Small_LLM_Model, functions: List[Functions]) \
            -> None:
        """
        Initializes the tokenizer-aware function routing mask.

        Builds token-to-id mappings from a dummy model encoding and creates
        scoring masks used to constrain token generation into categories such
        as numbers, characters, booleans, and function names.

        Args:
            model: Small language model used for tokenization and logits.
            functions: Collection of available callable functions.
        """
        self.model: Small_LLM_Model = model
        self.functions: List[Functions] = functions

        dummy = model.encode("hello").tolist()[0]
        dummy_logits: List[float] = model.get_logits_from_input_ids(dummy)
        self.id2tok: Dict[int, str] = {}
        for id, _ in enumerate(dummy_logits):
            self.id2tok[id] = model.decode([id])

        self.mvs: int = len(dummy_logits)

        self.M_numbers: npt.NDArray[np.float32] = np.full(
            self.mvs, -np.inf, dtype=np.float32
        )
        self.M_chars: npt.NDArray[np.float32] = np.full(
            self.mvs, -np.inf, dtype=np.float32
        )
        self.M_bool: npt.NDArray[np.float32] = np.full(
            self.mvs, -np.inf, dtype=np.float32
        )
        self.M_fun_name: npt.NDArray[np.float32] = np.full(
            self.mvs, -np.inf, dtype=np.float32
        )

        ids: Set[int] = set()
        names: List[Any] = [self.model.encode(f.name) for f in self.functions]
        for f in names:
            ids.update(f[0].tolist())
        self.M_fun_name[list(ids)] = 0.0

        for id, text in self.id2tok.items():
            clean_chars: str = text.replace("Ġ", " ").replace("Ċ", "\n")
            clean: str = text.replace("Ġ", "").replace("Ċ", "")

            if not clean_chars:
                continue

            if "\n" not in clean_chars:
                self.M_chars[id] = 0.0

            if not clean:
                continue

            if clean == '"':
                self.M_fun_name[id] = 0.0

            if all(c in "0123456789.-," for c in clean):
                if sum([1 for c in clean if c in ",.-"]) > 1:
                    continue
                self.M_numbers[id] = 0.0
