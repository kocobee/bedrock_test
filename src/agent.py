import os
import json
import httpx
import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "us.anthropic.claude-opus-4-8-20251101-v1:0"
USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"

SYSTEM_PROMPT = """You are a helpful customer support agent for Acme Corp. You assist customers with:
- Order status and tracking
- Product questions and troubleshooting
- Returns and refunds
- Account issues
- General FAQs

Be friendly, concise, and professional. Use the available tools to look up information when needed.
If an issue requires human escalation, use the escalate_to_human tool."""

tools = [
    {
        "name": "get_order_status",
        "description": "Look up the status of a customer order by order ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID to look up (e.g. ORD-12345)"
                }
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "get_product_info",
        "description": "Get information about a product including specs, availability, and pricing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "The name or SKU of the product"
                }
            },
            "required": ["product_name"]
        }
    },
    {
        "name": "submit_return_request",
        "description": "Submit a return or refund request for an order.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID to return"},
                "reason": {"type": "string", "description": "Reason for the return"}
            },
            "required": ["order_id", "reason"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Escalate the conversation to a human agent when the issue is too complex or the customer requests it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of the issue for the human agent"
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Priority level of the escalation"
                }
            },
            "required": ["summary", "priority"]
        }
    }
]


class _BedrockMockTransport(httpx.BaseTransport):
    """Intercepts Bedrock HTTP requests and returns a canned Anthropic-format response."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        try:
            body = json.loads(request.content)
            messages = body.get("messages", [])
            last = messages[-1]["content"] if messages else ""
            if isinstance(last, list):
                last_text = " ".join(b.get("text", "") for b in last if b.get("type") == "text")
            else:
                last_text = str(last)
        except Exception:
            last_text = ""

        last_lower = last_text.lower()
        if "order" in last_lower:
            text = "[MOCK BEDROCK] I can help with your order. Please share the order ID."
        elif "return" in last_lower or "refund" in last_lower:
            text = "[MOCK BEDROCK] I can help with a return. Please provide your order ID and reason."
        else:
            text = "[MOCK BEDROCK] Hello! I'm the Acme Corp support agent. How can I help you today?"

        return httpx.Response(
            200,
            json={
                "id": "msg_mock_bedrock",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "model": MODEL,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": len(text.split())},
            },
        )


def _make_client() -> anthropic.AnthropicBedrock:
    if USE_MOCK:
        return anthropic.AnthropicBedrock(
            aws_access_key="mock-key",
            aws_secret_key="mock-secret",
            aws_region="us-east-1",
            http_client=httpx.Client(transport=_BedrockMockTransport()),
        )
    return anthropic.AnthropicBedrock()


client = _make_client()


def handle_tool_call(tool_name: str, tool_input: dict) -> str:
    """Simulate tool execution — replace with real integrations."""
    if tool_name == "get_order_status":
        order_id = tool_input["order_id"]
        return json.dumps({
            "order_id": order_id,
            "status": "shipped",
            "estimated_delivery": "2026-06-14",
            "tracking_number": "1Z999AA10123456784",
            "carrier": "UPS"
        })

    elif tool_name == "get_product_info":
        product = tool_input["product_name"]
        return json.dumps({
            "product": product,
            "in_stock": True,
            "price": "$49.99",
            "description": f"High-quality {product} from Acme Corp.",
            "warranty": "1 year"
        })

    elif tool_name == "submit_return_request":
        return json.dumps({
            "return_id": "RET-98765",
            "order_id": tool_input["order_id"],
            "status": "approved",
            "instructions": "A prepaid shipping label will be emailed within 24 hours."
        })

    elif tool_name == "escalate_to_human":
        return json.dumps({
            "escalation_id": "ESC-11223",
            "status": "queued",
            "estimated_wait": "5-10 minutes",
            "summary": tool_input["summary"],
            "priority": tool_input["priority"]
        })

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def run_agent(conversation_history: list[dict]) -> str:
    """Run one agent turn, handling tool use loops, and return the assistant's final text."""
    messages = conversation_history.copy()

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason == "end_turn":
            for block in assistant_content:
                if block.type == "text":
                    return block.text
            return ""

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    print(f"  [tool: {block.name}({json.dumps(block.input)})]")
                    result = handle_tool_call(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})
            continue

        break

    return ""


def main():
    print("Acme Corp Customer Support")
    print("Type 'quit' or 'exit' to end the session.\n")

    conversation_history: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Thank you for contacting Acme Corp support. Goodbye!")
            break

        conversation_history.append({"role": "user", "content": user_input})

        response = run_agent(conversation_history)
        conversation_history.append({"role": "assistant", "content": response})

        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    main()
