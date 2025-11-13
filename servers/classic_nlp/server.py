from typing import Any, Dict, List
from fastapi import FastAPI
from pydantic import BaseModel
from pymystem3 import Mystem

mystem = Mystem()

app = FastAPI(title="Classic NLP")

TOOLS = {
    "lemmatize": {
        "description": "Лемматизация токенов с помощью Mystem",
        "params_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    },
    "stem": {
        "description": "Стемминг токенов через Mystem",
        "params_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    },
}

class RpcRequest(BaseModel):
    jsonrpc: str
    method: str
    params: Dict[str, Any]
    id: str

@app.post("/rpc")
async def rpc(req: RpcRequest) -> Dict[str, Any]:
    method = req.method
    params = req.params
    req_id = req.id

    if method == "list_tools":
        result = [
            {"name": name, "description": spec["description"], "params_schema": spec["params_schema"]}
            for name, spec in TOOLS.items()
        ]
        return {"jsonrpc": "2.0", "result": result, "id": req_id}
    
    if method == "call_tool":
        tool_name = params.get("tool_name")
        tool_params = params.get("params", {})
        text: str = tool_params["text"]

        if tool_name == 'lemmatize':
            result = mystem.lemmatize(text)
        elif tool_name == 'stem':
            analysis = mystem.analyze(text)
            stems = []
            for token in analysis:
                if 'analysis' in token and token['analysis']:
                    stems.append(token['analysis'][0]['lex'])
                else:
                    stems.append(token.get("text", ""))
            result = stems
        else:
            return {"jsonrpc": "2.0", "error": {"code": -32601, "message": "Unknown tool"}, "id": req_id}

        return {"jsonrpc": "2.0", "result": result, "id": req_id}

    return {"jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found"}, "id": req_id}