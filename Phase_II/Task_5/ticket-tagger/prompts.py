import json

CATEGORIES = [
    "Billing Issue",
    "Technical Support",
    "Account Access",
    "Network Issue",
    "Customer Complaint",
    "Refund Request",
    "Product Inquiry",
    "Escalation Required",
    "Shipping & Delivery",
    "Security Concern"
]

FEW_SHOT_EXAMPLES = [
    {
        "ticket": "I've been charged twice for my subscription this month and nobody is responding to my emails.",
        "tags": ["Billing Issue", "Customer Complaint", "Escalation Required"]
    },
    {
        "ticket": "I can't log into my account. I reset my password but still getting an error.",
        "tags": ["Account Access", "Technical Support", "Security Concern"]
    },
    {
        "ticket": "My internet has been down for 3 days. I work from home and this is affecting my livelihood.",
        "tags": ["Network Issue", "Customer Complaint", "Escalation Required"]
    },
    {
        "ticket": "I'd like to know the difference between your basic and premium plans.",
        "tags": ["Product Inquiry", "Billing Issue", "Customer Complaint"]
    },
    {
        "ticket": "My package was supposed to arrive 5 days ago and tracking shows it's still in transit.",
        "tags": ["Shipping & Delivery", "Customer Complaint", "Refund Request"]
    }
]


def zero_shot_prompt(ticket: str) -> str:
    categories_str = "\n".join(f"- {c}" for c in CATEGORIES)
    return f"""You are a support ticket classification system.

Given the following support ticket, return the top 3 most relevant tags from the list below.
Only return a valid JSON array of exactly 3 strings. No explanation, no extra text.

Available tags:
{categories_str}

Ticket:
\"\"\"{ticket}\"\"\"

Response format example: ["Tag1", "Tag2", "Tag3"]"""


def few_shot_prompt(ticket: str) -> str:
    categories_str = "\n".join(f"- {c}" for c in CATEGORIES)

    examples_str = ""
    for ex in FEW_SHOT_EXAMPLES:
        tags_str = json.dumps(ex["tags"])
        examples_str += f'Ticket: """{ex["ticket"]}"""\nTags: {tags_str}\n\n'

    return f"""You are a support ticket classification system.

Given a support ticket, return the top 3 most relevant tags from the list below.
Only return a valid JSON array of exactly 3 strings. No explanation, no extra text, no nested arrays.

Available tags:
{categories_str}

Examples:
{examples_str}Now classify this ticket:
Ticket: \"\"\"{ticket}\"\"\"
Tags:"""