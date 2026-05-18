from enum import Enum, auto
from grammar import JSONCharDFA, State as JSONState

class State(Enum):
    START = auto()
    OPEN_BRACE = auto()
    IN_NAME_KEY = auto()
    AFTER_NAME_KEY = auto()
    AFTER_NAME_COLON = auto()
    IN_NAME_STRING = auto()
    AFTER_NAME_STRING = auto()
    AFTER_NAME_COMMA = auto()
    IN_PARAMETERS_KEY = auto()
    AFTER_PARAMETERS_KEY = auto()
    AFTER_PARAMETERS_COLON = auto()
    IN_PARAMETERS_OBJECT = auto()
    AFTER_PARAMETERS_OBJECT = auto()
    END = auto()
    ERROR = auto()

class FunctionCallDFA:
    def __init__(self, function_names: list[str]):
        self.state = State.START
        self.function_names = function_names
        self.buffer = ""
        self.inner_dfa = None

    def clone(self):
        new = FunctionCallDFA(self.function_names)
        new.state = self.state
        new.buffer = self.buffer
        new.inner_dfa = self.inner_dfa.clone() if self.inner_dfa else None
        return new

    def _skip_whitespace(self, ch: str) -> bool:
        # Skip whitespace outside strings
        return ch in ' \t\n\r' and self.state not in (State.IN_NAME_STRING, State.IN_PARAMETERS_OBJECT)

    def process_char(self, ch: str) -> bool:
        if self._skip_whitespace(ch):
            return True

        if self.state == State.START:
            if ch == '{':
                self.state = State.OPEN_BRACE
                return True
            return False

        elif self.state == State.OPEN_BRACE:
            if ch == '"':
                self.state = State.IN_NAME_KEY
                self.buffer = ""
                return True
            return False

        elif self.state == State.IN_NAME_KEY:
            if ch == '"':
                if self.buffer == "name":
                    self.state = State.AFTER_NAME_KEY
                    return True
                return False
            self.buffer += ch
            return True

        elif self.state == State.AFTER_NAME_KEY:
            if ch == ':':
                self.state = State.AFTER_NAME_COLON
                return True
            return False

        elif self.state == State.AFTER_NAME_COLON:
            if ch == '"':
                self.state = State.IN_NAME_STRING
                self.buffer = ""
                return True
            return False

        elif self.state == State.IN_NAME_STRING:
            if ch == '"':
                if self.buffer in self.function_names:
                    self.state = State.AFTER_NAME_STRING
                    return True
                return False
            self.buffer += ch
            return True

        elif self.state == State.AFTER_NAME_STRING:
            if ch == ',':
                self.state = State.AFTER_NAME_COMMA
                return True
            return False

        elif self.state == State.AFTER_NAME_COMMA:
            if ch == '"':
                self.state = State.IN_PARAMETERS_KEY
                self.buffer = ""
                return True
            return False

        elif self.state == State.IN_PARAMETERS_KEY:
            if ch == '"':
                if self.buffer == "parameters":
                    self.state = State.AFTER_PARAMETERS_KEY
                    return True
                return False
            self.buffer += ch
            return True

        elif self.state == State.AFTER_PARAMETERS_KEY:
            if ch == ':':
                self.state = State.AFTER_PARAMETERS_COLON
                return True
            return False

        elif self.state == State.AFTER_PARAMETERS_COLON:
            if ch == '{':
                self.inner_dfa = JSONCharDFA(max_depth=10)
                if not self.inner_dfa.process_char(ch):
                    return False
                self.state = State.IN_PARAMETERS_OBJECT
                return True
            return False

        elif self.state == State.IN_PARAMETERS_OBJECT:
            if not self.inner_dfa.process_char(ch):
                return False
            if self.inner_dfa.state == JSONState.END:
                self.state = State.AFTER_PARAMETERS_OBJECT
            return True

        elif self.state == State.AFTER_PARAMETERS_OBJECT:
            if ch == '}':
                self.state = State.END
                return True
            return False

        elif self.state == State.END:
            return False

        return False