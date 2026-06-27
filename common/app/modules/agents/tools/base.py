"""Agent tools: Pydantic-validated inputs and OpenAI/Gemini JSON schemas."""

from __future__ import annotations

import textwrap
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from google.genai.types import (
    FunctionDeclaration as GeminiFunctionDeclaration,
)
from google.genai.types import (
    JSONSchema as GeminiJSONSchema,
)
from google.genai.types import (
    Schema as GeminiSchema,
)
from google.genai.types import (
    Tool as GeminiTool,
)
from google.genai.types import (
    Type as GeminiType,
)
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition
from pydantic import BaseModel

from common.app.modules.llm.messages import ToolResultContentBlock

BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


class ToolResult(BaseModel):
    content: str
    is_error: bool = False
    break_out_of_loop: bool = False


class ImageToolResult(BaseModel):
    content: list[ToolResultContentBlock]
    is_error: bool = False
    break_out_of_loop: bool = False


class Tool(ABC, Generic[BaseModelT]):
    name: str
    input_model: type[BaseModelT]
    human_in_the_loop: bool = False
    timeout: int = 30
    _current_tool_id: str | None = None

    def __init__(
        self,
        name: str,
        input_model: type[BaseModelT],
        *,
        human_in_the_loop: bool = False,
        timeout: int | None = 30,
    ) -> None:
        self.name = name
        self.input_model = input_model
        self.human_in_the_loop = human_in_the_loop
        self.timeout = 30 if timeout is None else timeout
        self._current_tool_id = None

    @property
    def description(self) -> str:
        return textwrap.dedent(self.input_model.__doc__ or "")

    def to_openai_tool(self) -> ChatCompletionToolParam:
        return ChatCompletionToolParam(
            type="function",
            function=FunctionDefinition(
                name=self.name,
                description=self.description,
                parameters=self.input_model.model_json_schema(),
            ),
        )

    def to_gemini_tool(self) -> GeminiTool:
        return GeminiTool(
            function_declarations=[
                GeminiFunctionDeclaration(
                    name=self.name,
                    description=self.description,
                    parameters=GeminiSchema.from_json_schema(
                        json_schema=GeminiJSONSchema.model_validate(
                            self.input_model.model_json_schema()
                        ),
                        raise_error_on_unsupported_field=False,
                    ),
                    response=GeminiSchema(type=GeminiType.STRING),
                )
            ]
        )

    @abstractmethod
    async def execute(self, input: BaseModelT) -> str | ToolResult | ImageToolResult:
        raise NotImplementedError

    async def cancel(self, input: BaseModelT) -> str:
        return "cancelled"
