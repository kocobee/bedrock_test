import anthropic
import json
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-opus-4-8"

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


def handle_tool_call(tool_name: str, tool_input: dict) -> str:
    """Simulate tool execution — replace with real integrations."""
    if tool_name == "get_order_status":
        order_id = tool_input["order_id"]
        # Simulated response
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

        # Collect assistant content
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason == "end_turn":
            # Extract text from content blocks
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

        # Unexpected stop reason
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
