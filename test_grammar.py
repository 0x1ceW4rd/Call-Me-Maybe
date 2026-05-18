# test_grammar.py
from src.grammar import load_vocab, JSONGrammarValidator

def main():
    # Load the real vocabulary from the model (uses Small_LLM_Model inside)
    vocab = load_vocab()                     # dict: token_str -> token_id
    validator = JSONGrammarValidator(vocab)

    # A few partial outputs to test
    test_cases = [
        "",                     # start: whitespace, {, [ allowed
        "{",                    # inside object: key string or } allowed
        '{"',                   # inside key: any string characters or closing "
        '{"name"',              # after key: : expected
        '{"name":',             # after colon: value start expected
        '{"name": "John"',      # after value: , or } expected
        '{"name": "John",',     # after comma: new key string expected
        '{"name": "John"}',      # complete object: only whitespace
    ]

    for partial in test_cases:
        print(f"Partial: {partial!r}")
        allowed_ids = validator.get_valid_tokens(partial)
        # Show the first 10 allowed token strings for readability
        allowed_strs = [t for t, i in vocab.items() if i in allowed_ids]
        print(f"  Allowed tokens ({len(allowed_ids)}): {allowed_strs[:10]}...\n")

if __name__ == "__main__":
    main()