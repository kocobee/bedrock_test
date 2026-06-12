"""AgentMark executor for the Acme Corp support agent.

The executor is the single seam where AgentMark's neutral prompt render
(``{messages, text_config}``) is handed to our LLM SDK. We reuse the app's
existing mock-aware AWS Bedrock invoke helper from ``src/agent.py`` so runs work
with ``USE_MOCK=true`` and no AWS credentials, and so AgentMark traces the same
model call the app actually makes.
"""

from agentmark.prompt_core import create_executor, ExecutorTextResult, UsageData

from src.agent import invoke_model


def _text(formatted, ctx) -> ExecutorTextResult:
    # ``model_name`` is a registry id like
    # "bedrock/us.anthropic.claude-opus-4-8-20251101-v1:0". Strip the provider
    # prefix to get the Bedrock model id InvokeModel expects.
    model_id = formatted.text_config.model_name.split("/", 1)[-1]

    messages = [m.model_dump(exclude_none=True) for m in formatted.messages]
    # Bedrock's Anthropic API takes ``system`` as a top-level field, so split
    # any system messages out of the neutral render.
    system = "\n".join(
        m["content"]
        for m in messages
        if m["role"] == "system" and isinstance(m["content"], str)
    )
    chat = [m for m in messages if m["role"] != "system"]

    res = invoke_model(
        chat,
        model=model_id,
        max_tokens=formatted.text_config.max_tokens or 1024,
        system=system or None,
    )
    text = "".join(b["text"] for b in res["content"] if b["type"] == "text")
    return ExecutorTextResult(
        text=text,
        usage=UsageData(
            input_tokens=res["usage"]["input_tokens"],
            output_tokens=res["usage"]["output_tokens"],
        ),
    )


executor = create_executor(name="bedrock-anthropic", text=_text)
