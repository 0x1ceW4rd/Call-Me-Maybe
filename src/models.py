from pydantic import BaseModel
from typing import Dict, Any

class FunctionDefinition(BaseModel):
    name: str
    description: str = ""
    parameters: Dict[str, Dict[str, str]]
    returns: Dict[str, str] = {}

class TestCase(BaseModel):
    prompt: str

class Result(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]