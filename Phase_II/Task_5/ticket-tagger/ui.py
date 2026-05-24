import os
import json
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from prompts import zero_shot_prompt, few_shot_prompt, CATEGORIES

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Ticket Tagger", page_icon="🎫", layout="centered")

st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 2.5rem 2rem !important; max-width: 800px !important; }

    .stApp { background-color: #111111; color: #e0e0e0; }

    [data-testid="stSidebar"] { display: none; }

    h1 { font-size: 22px !important; font-weight: 600 !important; color: #ffffff !important; letter-spacing: -0.3px; }
    p, label, .stCaption { color: #888 !important; font-size: 13px !important; }

    .stTextArea textarea {
        background-color: #1c1c1c !important;
        color: #e0e0e0 !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 8px !important;
        font-size: 14px !important;
    }
    .stTextArea textarea:focus {
        border-color: #444 !important;
        box-shadow: none !important;
    }
    .stTextArea textarea::placeholder { color: #444 !important; }

    .stSelectbox > div > div {
        background-color: #1c1c1c !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 8px !important;
        color: #e0e0e0 !important;
        font-size: 13px !important;
    }

    .stButton > button {
        background-color: #ffffff !important;
        color: #111111 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 20px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background-color: #e0e0e0 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid #2a2a2a !important;
        gap: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #555 !important;
        font-size: 13px !important;
        padding: 8px 16px !important;
        border-radius: 0 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 2px solid #ffffff !important;
        background: transparent !important;
    }

    .ticket-box {
        background: #1c1c1c;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 14px 16px;
        font-size: 14px;
        color: #ccc;
        margin-bottom: 20px;
        line-height: 1.6;
    }
    .section-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #555 !important;
        margin-bottom: 8px;
    }
    .tag-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 16px;
    }
    .tag {
        padding: 5px 12px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    .tag-zero { background: #1e1e1e; color: #d0d0d0; border: 1px solid #333; }
    .tag-few  { background: #242424; color: #d0d0d0; border: 1px solid #3a3a3a; }
    .tag-both { background: #2a2a2a; color: #ffffff; border: 1px solid #444; }

    .divider { border: none; border-top: 1px solid #222; margin: 20px 0; }

    .result-wrapper {
        background: #161616;
        border: 1px solid #222;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .expander-ticket {
        font-size: 13px;
        color: #aaa;
        margin-bottom: 12px;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


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
        if isinstance(tags, list):
            return [t for t in tags if isinstance(t, str)][:3]
        return [raw]
    except json.JSONDecodeError:
        return [raw]


def render_tags(tags, cls):
    return "<div class='tag-row'>" + "".join(
        f"<span class='tag {cls}'>{t}</span>" for t in tags
    ) + "</div>"


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Support Ticket Tagger")
st.caption("Zero-shot and few-shot LLM classification · GPT-3.5")
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Single", "Batch"])

# ── Tab 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    ticket_input = st.text_area(
        "Ticket",
        placeholder="Paste a support ticket here...",
        height=120,
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        mode = st.selectbox(
            "Mode",
            ["Both", "Zero-Shot Only", "Few-Shot Only"],
            label_visibility="collapsed"
        )
    with col2:
        run = st.button("Tag", use_container_width=True)

    if run and ticket_input.strip():
        with st.spinner(""):
            zero_tags, few_tags = [], []
            if mode in ["Both", "Zero-Shot Only"]:
                zero_tags = tag_ticket(zero_shot_prompt, ticket_input)
            if mode in ["Both", "Few-Shot Only"]:
                few_tags = tag_ticket(few_shot_prompt, ticket_input)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown(f"<div class='ticket-box'>📩 {ticket_input}</div>", unsafe_allow_html=True)

        if mode == "Both":
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("<div class='section-label'>Zero-Shot</div>", unsafe_allow_html=True)
                st.markdown(render_tags(zero_tags, "tag-zero"), unsafe_allow_html=True)
            with col_b:
                st.markdown("<div class='section-label'>Few-Shot</div>", unsafe_allow_html=True)
                st.markdown(render_tags(few_tags, "tag-few"), unsafe_allow_html=True)

            overlap = list(set(zero_tags) & set(few_tags))
            if overlap:
                st.markdown("<div class='section-label'>Agreed on both</div>", unsafe_allow_html=True)
                st.markdown(render_tags(overlap, "tag-both"), unsafe_allow_html=True)

        elif mode == "Zero-Shot Only":
            st.markdown("<div class='section-label'>Zero-Shot</div>", unsafe_allow_html=True)
            st.markdown(render_tags(zero_tags, "tag-zero"), unsafe_allow_html=True)
        else:
            st.markdown("<div class='section-label'>Few-Shot</div>", unsafe_allow_html=True)
            st.markdown(render_tags(few_tags, "tag-few"), unsafe_allow_html=True)

    elif run:
        st.warning("Enter a ticket first.")

# ── Tab 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    batch_input = st.text_area(
        "Tickets",
        placeholder="One ticket per line...",
        height=180,
        label_visibility="collapsed"
    )
    batch_run = st.button("Tag All", use_container_width=False)

    if batch_run and batch_input.strip():
        tickets = [t.strip() for t in batch_input.strip().split("\n") if t.strip()]
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        for i, ticket in enumerate(tickets):
            with st.spinner(f"Ticket {i+1}/{len(tickets)}"):
                zero_tags = tag_ticket(zero_shot_prompt, ticket)
                few_tags  = tag_ticket(few_shot_prompt, ticket)

            with st.expander(f"#{i+1} · {ticket[:70]}..."):
                st.markdown(f"<div class='expander-ticket'>{ticket}</div>", unsafe_allow_html=True)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("<div class='section-label'>Zero-Shot</div>", unsafe_allow_html=True)
                    st.markdown(render_tags(zero_tags, "tag-zero"), unsafe_allow_html=True)
                with col_b:
                    st.markdown("<div class='section-label'>Few-Shot</div>", unsafe_allow_html=True)
                    st.markdown(render_tags(few_tags, "tag-few"), unsafe_allow_html=True)

    elif batch_run:
        st.warning("Enter at least one ticket.")