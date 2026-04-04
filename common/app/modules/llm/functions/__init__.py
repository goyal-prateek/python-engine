import inspect
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    Union,
    overload,
)

from pydantic import BaseModel, Field

from common.app.modules.llm.agent_completion import (
    AgentModelConfig,
    complete_llm_prompt_items,
    stream_llm_prompt_text_chunks,
)
from common.app.modules.llm.clients import SharedLLMClients
from common.app.modules.llm.promtps import LLMPromptItem


class LLMFunction(BaseModel):
    class ParamsModel(BaseModel):
        pass

    class OutputModel(BaseModel):
        pass

    class Step(BaseModel):
        type: Literal["transform", "completion"]

    class TransformStep(Step):
        type: Literal["transform"] = "transform"
        function: Callable[[Any], Dict[str, Any]]

        def execute(self, step_input: Any) -> Dict[str, Any]:
            return self.function(step_input)

    class CompletionStep(Step):
        type: Literal["completion"] = "completion"
        model: str
        extend_prompt: Optional[
            Callable[[Any, "LLMFunction.ParamsModel"], List[LLMPromptItem]]
        ] = None
        max_tokens: Optional[int] = None
        temperature: Optional[float] = None
        agent_model_config: Optional[AgentModelConfig] = Field(
            default=None,
            description="If set, sent to CompletionRouter as-is; else built from model / max_tokens / temperature with provider 'openai' (OpenRouter).",
        )

        def resolved_model_config(self) -> AgentModelConfig:
            if self.agent_model_config is not None:
                return self.agent_model_config.model_copy(deep=True)
            return AgentModelConfig(
                provider="openai",
                model=self.model,
                max_tokens=self.max_tokens
                if self.max_tokens is not None
                else 8192,
                temperature=0.7 if self.temperature is None else self.temperature,
            )

        @overload
        async def execute(
            self,
            step_input: Any,
            params: "LLMFunction.ParamsModel",
            shared_clients: SharedLLMClients,
            stream: Literal[False] = False,
        ) -> Optional[str]: ...

        @overload
        async def execute(
            self,
            step_input: Any,
            params: "LLMFunction.ParamsModel",
            shared_clients: SharedLLMClients,
            stream: Literal[True] = True,
        ) -> AsyncIterator[str]: ...

        async def execute(
            self,
            step_input: Any,
            params: "LLMFunction.ParamsModel",
            shared_clients: SharedLLMClients,
            stream: bool = False,
        ) -> Union[Optional[str], AsyncIterator[str]]:
            prompt: List[LLMPromptItem] = []
            if self.extend_prompt is not None:
                if inspect.iscoroutinefunction(self.extend_prompt):
                    prompt = await self.extend_prompt(step_input, params)
                else:
                    prompt = self.extend_prompt(step_input, params)

            cfg = self.resolved_model_config()
            if stream:
                return stream_llm_prompt_text_chunks(
                    shared_clients,
                    prompt=prompt,
                    system="",
                    config=cfg,
                )
            msg = await complete_llm_prompt_items(
                shared_clients,
                prompt=prompt,
                system="",
                config=cfg,
            )
            return msg.text_content or None

    name: str = Field(..., title="Name", description="Name of the function")
    steps: List[Union[TransformStep, CompletionStep]] = Field(
        [], description="Steps to execute"
    )
    params_model: Type[ParamsModel] = Field(..., description="Parameters model")
    output_model: Optional[Type[OutputModel]] = Field(default=None, description="Output model")
    stream: bool = Field(default=False, description="Stream the output - only the last completion step can be streamed")

    async def _stream_last_completion(
        self,
        step: CompletionStep,
        step_input: Any,
        params: ParamsModel,
        shared_clients: SharedLLMClients,
    ) -> AsyncGenerator[str, None]:
        agen = await step.execute(step_input, params, shared_clients, stream=True)
        async for chunk in agen:
            if chunk is not None:
                yield chunk

    async def run(
        self,
        params: ParamsModel,
        *,
        shared_llm_clients: SharedLLMClients,
    ) -> Union[OutputModel, Any, AsyncGenerator[str, None]]:
        intermediate_outputs: List[Any] = []
        if self.stream and not isinstance(self.steps[-1], LLMFunction.CompletionStep):
            raise ValueError(
                "Last step must be a completion step when stream is True"
            )

        for index, step in enumerate(self.steps):
            step_output: Optional[Dict[str, Any]] = None
            step_input = (
                intermediate_outputs[-1] if len(intermediate_outputs) > 0 else params
            )

            if isinstance(step, LLMFunction.TransformStep):
                step_output = step.execute(step_input)
                intermediate_outputs.append(step_output)

            elif isinstance(step, LLMFunction.CompletionStep):
                if self.stream and index == len(self.steps) - 1:
                    return self._stream_last_completion(
                        step, step_input, params, shared_llm_clients
                    )

                completion = await step.execute(
                    step_input, params, shared_llm_clients, stream=False
                )

                if completion is not None:
                    intermediate_outputs.append(completion)

        if self.output_model is not None:
            return self.output_model.model_validate(intermediate_outputs[-1])
        else:
            return intermediate_outputs[-1]