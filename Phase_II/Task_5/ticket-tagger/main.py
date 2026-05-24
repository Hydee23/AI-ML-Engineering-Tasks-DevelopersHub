import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from prompts import zero_shot_prompt, few_shot_prompt

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Sample support tickets ────────────────────────────────────────────────────
tickets = [
    "I was charged $99 but I only signed up for the $49 plan. Please refund the difference.",
    "My app keeps crashing every time I try to open it on my iPhone.",
    "Someone logged into my account from a different country. I need this investigated immediately.",
    "The wifi router you sent me is not working. I've tried resetting it multiple times.",
    "I've been waiting 3 weeks for my order. The tracking page says delivered but I got nothing.",
    "Can you explain what features come with the enterprise plan?",
    "I want to cancel my subscription and get a full refund for this month.",
    "Your customer service rep was extremely rude to me on the call yesterday.",
    "I forgot my password and the reset email is not arriving.",
    "My connection keeps dropping every 30 minutes. I've already restarted the router."
]


def tag_ticket(prompt_fn, ticket: str) -> list:
    prompt = prompt_fn(ticket)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw = response.choices[0].message.content.strip()
    try:
        tags = json.loads(raw)
        return tags[:3]
    except json.JSONDecodeError:
        return [raw]


# ── Run both approaches ───────────────────────────────────────────────────────
print("Running zero-shot and few-shot tagging...\n")

results = []
for i, ticket in enumerate(tickets):
    print(f"Ticket {i+1}: {ticket[:60]}...")

    zero_tags = tag_ticket(zero_shot_prompt, ticket)
    few_tags  = tag_ticket(few_shot_prompt, ticket)

    print(f"  Zero-shot : {zero_tags}")
    print(f"  Few-shot  : {few_tags}\n")

    results.append({
        "ticket": ticket,
        "zero_shot_tags": ", ".join(zero_tags),
        "few_shot_tags": ", ".join(few_tags),
        "tags_match": zero_tags == few_tags
    })

# ── Save results ──────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("results/output.csv", index=False)
print("Results saved to results/output.csv")

# ── Summary ───────────────────────────────────────────────────────────────────
match_count = df["tags_match"].sum()
print(f"\nSummary:")
print(f"  Total tickets     : {len(tickets)}")
print(f"  Exact tag matches : {match_count}/{len(tickets)}")
print(f"  Agreement rate    : {match_count/len(tickets)*100:.1f}%")