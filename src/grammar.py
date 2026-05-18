import json
from enum import Enum, auto

def load_vocab() -> dict[str, int]:
    from llm_sdk import Small_LLM_Model
    path = Small_LLM_Model().get_path_to_vocab_file()
    with open(path) as f:
        return json.load(f)


class State(Enum):
    START = auto()
    OBJECT_START = auto()
    KEY = auto()
    KEY_END = auto()
    COLON = auto()
    VALUE_START = auto()
    STRING = auto()
    STRING_ESCAPE = auto()
    NUMBER = auto()
    LITERAL = auto()
    AFTER_VALUE = auto()
    COMMA = auto()
    END = auto()
    ERROR = auto()


class JSONCharDFA:
    def __init__(self, max_depth=10):
        self.stack = []
        self.state = State.START
        self.escape = False
        self.literal_buf = ''
        self.max_depth = max_depth

    def clone(self):
        new = JSONCharDFA(self.max_depth)
        new.stack = self.stack.copy()
        new.state = self.state
        new.escape = self.escape
        new.literal_buf = self.literal_buf
        return new

    def _start_value(self, ch: str) -> bool:
        if ch == '"':
            self.state = State.STRING
            self.escape = False
            return True
        if ch == '-' or ch.isdigit():
            self.state = State.NUMBER
            return True
        if ch == 't':
            self.literal_buf = 't'
            self.state = State.LITERAL
            return True
        if ch == 'f':
            self.literal_buf = 'f'
            self.state = State.LITERAL
            return True
        if ch == 'n':
            self.literal_buf = 'n'
            self.state = State.LITERAL
            return True
        if ch == '{':
            if len(self.stack) >= self.max_depth:
                return False
            self.stack.append('}')
            self.state = State.OBJECT_START
            return True
        if ch == '[':
            if len(self.stack) >= self.max_depth:
                return False
            self.stack.append(']')
            self.state = State.AFTER_VALUE
            return True
        return False

    def process_char(self, ch: str) -> bool:
        if ch in ' \t\n\r' and self.state not in (State.STRING, State.STRING_ESCAPE):
            return True

        if self.state == State.START:
            if ch == '{':
                self.stack.append('}')
                self.state = State.OBJECT_START
                return True
            if ch == '[':
                self.stack.append(']')
                self.state = State.AFTER_VALUE
                return True
            self.state = State.ERROR
            return False

        elif self.state == State.OBJECT_START:
            if ch == '"':
                self.state = State.KEY
                self.escape = False
                return True
            if ch == '}':
                if not self.stack or self.stack[-1] != '}':
                    self.state = State.ERROR
                    return False
                self.stack.pop()
                self.state = State.END if not self.stack else State.AFTER_VALUE
                return True
            self.state = State.ERROR
            return False

        elif self.state == State.KEY:
            if self.escape:
                self.escape = False
                return True
            if ch == '\\':
                self.escape = True
                return True
            if ch == '"':
                self.state = State.KEY_END
                return True
            return True

        elif self.state == State.KEY_END:
            if ch == ':':
                self.state = State.COLON
                return True
            self.state = State.ERROR
            return False

        elif self.state == State.COLON:
            return self._start_value(ch)

        elif self.state == State.STRING:
            if self.escape:
                self.escape = False
                return True
            if ch == '\\':
                self.escape = True
                return True
            if ch == '"':
                self.state = State.AFTER_VALUE
                return True
            return True

        elif self.state == State.NUMBER:
            if ch in '0123456789.eE+-':
                return True
            self.state = State.AFTER_VALUE
            return self.process_char(ch)

        elif self.state == State.LITERAL:
            valid_literals = ['true', 'false', 'null']
            possible = [lit for lit in valid_literals if lit.startswith(self.literal_buf + ch)]
            if possible:
                self.literal_buf += ch
                return True
            if self.literal_buf in valid_literals:
                self.state = State.AFTER_VALUE
                return self.process_char(ch)
            self.state = State.ERROR
            return False

        elif self.state == State.AFTER_VALUE:
            if not self.stack:
                if ch in ' \t\n\r':
                    self.state = State.END
                    return True
                self.state = State.ERROR
                return False
            expected_close = self.stack[-1]
            if ch == ',':
                self.state = State.COMMA
                return True
            if ch == expected_close:
                self.stack.pop()
                self.state = State.END if not self.stack else State.AFTER_VALUE
                return True
            self.state = State.ERROR
            return False

        elif self.state == State.COMMA:
            if not self.stack:
                self.state = State.ERROR
                return False
            if self.stack[-1] == '}':
                if ch == '"':
                    self.state = State.KEY
                    self.escape = False
                    return True
                self.state = State.ERROR
                return False
            if self.stack[-1] == ']':
                return self._start_value(ch)
            self.state = State.ERROR
            return False

        elif self.state == State.END:
            if ch in ' \t\n\r':
                return True
            self.state = State.ERROR
            return False

        else:
            self.state = State.ERROR
            return False


class JSONGrammarValidator:
    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.token_strs = list(vocab.keys())

    def get_valid_tokens(self, partial_output: str) -> list[int]:
        dfa = JSONCharDFA()
        for ch in partial_output:
            if not dfa.process_char(ch):
                return []
        allowed = []
        for token_str in self.token_strs:
            test_dfa = dfa.clone()
            ok = True
            for ch in token_str:
                if not test_dfa.process_char(ch):
                    ok = False
                    break
            if ok:
                allowed.append(self.vocab[token_str])
        return allowed