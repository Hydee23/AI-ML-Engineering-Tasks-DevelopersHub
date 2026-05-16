import streamlit as st
import re

def apply_styles():
    st.markdown("""
    <style>
        #MainMenu, footer { visibility: hidden; }
        .block-container { padding: 2rem 3rem !important; max-width: 850px !important; margin: 0 auto; }
        
        .msg-box {
            padding: 1rem 1.25rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            line-height: 1.6;
            font-size: 15px;
        }
        .user-msg {
            background-color: #1e1e1e;
            border-left: 4px solid #7f77dd;
            color: #f0f0f0;
        }
        .bot-msg {
            background-color: #141414;
            border-left: 4px solid #4ade80;
            color: #e0e0e0;
        }
        .msg-header {
            margin-bottom: 0.5rem;
            display: block;
        }

        [data-testid="stChatInputSubmitButton"],
        [data-testid="stChatInputSubmitButton"] button {
            background-color: #f0f0f0 !important;
            border-radius: 8px !important;
            border: none !important;
        }
        [data-testid="stChatInputSubmitButton"] svg {
            fill: #0f0f0f !important; 
        }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.markdown("<center><h2>Context-Aware Chatbot</h2></center>", unsafe_allow_html=True)
        st.divider()
        
        # Static Powered By Notice Label
        st.markdown("<p style='font-size: 14px; margin-bottom: 0; opacity: 0.7;'>Powered by:</p>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #4ade80; margin-top: 0; font-weight: 600;'>GPT-4o Mini</h4>", unsafe_allow_html=True)
        
        st.divider()
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    return "gpt-4o-mini"

def render_header(selected_model):
    st.markdown(f"")

def render_welcome():
    st.markdown("""
    <div class='msg-box bot-msg'>
        <strong class='msg-header'>Bee-MO</strong>
        <b>Hi! Ask me anything about AI, ML, NLP, or Neural Networks.</b>
    </div>
    """, unsafe_allow_html=True)

def render_messages():
    for msg in st.session_state.messages:
        raw_content = msg["content"]
        
        # 1. Parse markdown bold tokens (**text**) into HTML strong elements
        processed_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', raw_content)
        
        # 2. Sequential line compiler processes arrays inside a single markup context block
        html_lines = []
        for line in processed_content.split("\n"):
            stripped = line.strip()
            if not stripped:
                html_lines.append("<div style='height: 6px;'></div>")
                continue
            
            num_match = re.match(r'^(\d+)\.\s+(.*)', stripped)
            bullet_match = re.match(r'^([\-\*•])\s+(.*)', stripped)
            
            if num_match:
                html_lines.append(f"<div style='margin-left: 16px; margin-bottom: 4px;'>{num_match.group(1)}. {num_match.group(2)}</div>")
            elif bullet_match:
                html_lines.append(f"<div style='margin-left: 16px; margin-bottom: 4px;'>• {bullet_match.group(2)}</div>")
            else:
                html_lines.append(f"<div style='margin-bottom: 4px;'>{line}</div>")
                
        content_html = "".join(html_lines)

        if msg["role"] == "user":
            st.markdown(f"""
            <div class='msg-box user-msg'>
                <strong class='msg-header'>You:</strong>
                {content_html}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='msg-box bot-msg'>
                <strong class='msg-header'>Bee-MO:</strong>
                {content_html}
            </div>
            """, unsafe_allow_html=True)