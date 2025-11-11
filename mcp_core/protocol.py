from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

@dataclass
class Message:
    role: Role
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolSpec:
    name: str
    description: str
    params_schema: Dict[str, Any]

@dataclass
class ToolCall:
    tool_name: str
    params: Dict[str, Any]

@dataclass
class JsonRpcRequest:
    jsonrpc: str
    method: str
    params: Dict[str, Any]
    id: str

@dataclass
class JsonRpcResponse:
    jsonrpc: str
    result: Any
    id: str
    error: Optional[Dict[str, Any]] = None
