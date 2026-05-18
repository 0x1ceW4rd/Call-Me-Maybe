from typing import List, Tuple, Optional
from models import FunctionDefinition, Result

def validate_against_schema(
    result: Result,
    definitions: List[FunctionDefinition]
) -> Tuple[bool, Optional[str]]:
    func = next((f for f in definitions if f.name == result.name), None)
    if func is None:
        return False, f"Unknown function '{result.name}'"

    expected = set(func.parameters.keys())
    actual = set(result.parameters.keys())
    missing = expected - actual
    if missing:
        return False, f"Missing parameters: {missing}"
    extra = actual - expected
    if extra:
        return False, f"Extra parameters: {extra}"

    for param, schema in func.parameters.items():
        typ = schema.get("type")
        value = result.parameters.get(param)
        if value is None:
            return False, f"Parameter '{param}' missing"
        if typ == "number" and not isinstance(value, (int, float)):
            return False, f"Parameter '{param}' should be number, got {type(value).__name__}"
        if typ == "string" and not isinstance(value, str):
            return False, f"Parameter '{param}' should be string, got {type(value).__name__}"
        if typ == "boolean" and not isinstance(value, bool):
            return False, f"Parameter '{param}' should be boolean, got {type(value).__name__}"
    return True, None