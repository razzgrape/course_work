import json
import uuid
from typing import Any, Dict, Optional
import requests
from mcp_core.protocol import JsonRpcRequest, ToolSpec

class McpClient:
    def __init__(self, base_url: str, auth_token: Optional[str] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-type": "application/json"}
        if self.auth_token:
            headers['Authorization'] = f"Bearer {self.auth_token}"
        return headers
    
    def _rpc(self, method: str, params: Dict[str, Any]) -> Any:
        req = JsonRpcRequest(jsonrpc="2.0", method = method, params=params, id=str(uuid.uuid4()))
        resp = requests.post(f"{self.base_url}/rpc", headers=self._headers(), data=json.dumps(req.__dict__), timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        if "error" in payload and payload["error"]:
            raise RuntimeError(f'RPC error: {payload['error']}')
        return payload.get("result")
    
    def list_tool(self) -> Dict[str, ToolSpec]:
        result = self._rpc("list_tools", {})
        tools = {}
        for t in result:
            tools[t["name"]] = ToolSpec(name=t["name"], description=t["description"], params_schema=t["params_schema"])
        return tools
    
    def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        return self._rpc("call_tool", {"tool_name": tool_name, "params": params})
    
    def get_resourse(self, uri: str) -> Any:
        return self._rpc("get_resourse", {"uri": uri})
    