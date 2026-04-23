from pydantic import BaseModel
from typing import List, Optional, Any


class Parameter(BaseModel):
    type: str
    enum: Optional[List[Any]] = None

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: str


class JSONSchema:
    def __init__(self, functions: List[FunctionDefinition]):
        self.functions = functions