"""Muse Spark lab partner: a tool loop over the Responses API. Numbers come from tools; the model plans and explains."""

import json

from openai import OpenAI

from . import grounding
from .config import META_BASE_URL, api_key
from .tools import TOOL_FUNCS, TOOL_SCHEMAS

AGENT_MODEL = "muse-spark-1.3"
MAX_TURNS = 8

INSTRUCTIONS = """You are VideoLab's lab partner. VideoLab turns a phone slow-motion video into physics measurements:
SAM 3.1 tracks the object, an A4 sheet calibrates distances, and deterministic fits produce the numbers.

Rules:
- Never compute or estimate a number yourself. Every number you state must be copied from a tool result.
- Cite each measured value with its ref in square brackets right after it, e.g. "g = 9.76 ± 0.07 m/s² [r1.g]".
- Before interpreting a run, call check_quality on it, and mention any warn or fail checks.
- When the student made a prediction, start from it: what they predicted, what was measured, the gap, and the
  physical reason for the gap. Use explain_physics for the model, assumptions and error sources.
- Be concise: short paragraphs or bullets, no more than about 150 words unless asked for more.
- If something can't be measured from the data, say so plainly."""


class LabAgent:
    def __init__(self) -> None:
        self.client = OpenAI(base_url=META_BASE_URL, api_key=api_key())
        self.previous_response_id: str | None = None
        self.all_results: list[dict] = []  # tool results across the conversation, so follow-ups can reuse earlier numbers

    def ask(self, message: str) -> dict:
        tool_log: list[dict] = []
        response = self.client.responses.create(
            model=AGENT_MODEL,
            instructions=INSTRUCTIONS,
            input=message,
            tools=TOOL_SCHEMAS,
            previous_response_id=self.previous_response_id,
        )
        for _ in range(MAX_TURNS):
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                break
            outputs = []
            for call in calls:
                result = self._run_tool(call.name, call.arguments)
                tool_log.append({"tool": call.name, "arguments": call.arguments, "result": result})
                outputs.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(result)})
            response = self.client.responses.create(
                model=AGENT_MODEL,
                instructions=INSTRUCTIONS,
                input=outputs,
                tools=TOOL_SCHEMAS,
                previous_response_id=response.id,
            )
        self.previous_response_id = response.id
        self.all_results.extend(t["result"] for t in tool_log)
        answer = response.output_text
        return {
            "answer": answer,
            "tools": tool_log,
            "grounding": grounding.check(answer, self.all_results),
            "usage": response.usage.model_dump() if response.usage else None,
        }

    @staticmethod
    def _run_tool(name: str, arguments: str) -> dict:
        fn = TOOL_FUNCS.get(name)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**json.loads(arguments or "{}"))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
            return {"error": str(e)}
