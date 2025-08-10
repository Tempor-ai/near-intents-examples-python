# app.py
import os
from typing import List, TypedDict, Literal, Dict, Any, Union

import streamlit as st
from openai import OpenAI
from prompt import SYSTEM_PROMPT
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Streamlit + OpenAI Responses API", page_icon="💬")

# --- Sidebar config (unchanged look) ---
with st.sidebar:
    st.title("⚙️ Settings")
    # You can add GPT-5 models here later; logic below will auto-handle them
    model_list = ["gpt-4o", "gpt-4o-mini"]
    model = st.selectbox("Model", model_list, index=0)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Session state for chat history (Streamlit-native chat UI) ---
if "messages" not in st.session_state:
    # each item: {"role": "user"|"assistant", "content": "..." OR list[content blocks]}
    st.session_state.messages = []

st.title("NEAR Intents Swap Quote Assistant")

# Render the history (unchanged look)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        # Keep simple markdown rendering
        text = m["content"]
        if isinstance(text, str):
            st.markdown(text)
        else:
            # If some messages are multimodal blocks, show any text parts
            txt_parts = [c.get("text", "") for c in text if isinstance(c, dict) and c.get("type") in ("input_text", "summary_text")]
            st.markdown("\n\n".join([t for t in txt_parts if t]))

# Input box
user_input = st.chat_input("Type your message…")


# ---------- Helpers: types + formatting ----------

Role = Literal["user", "assistant", "system"]

class ChatMessage(TypedDict):
    role: Role
    content: Union[str, List[Dict[str, Any]]]

ALLOWED_INPUT_TYPES = {
    "input_text",
    "input_image",
    "input_file",
    "computer_screenshot",
    "summary_text",
}

def normalize_to_content_blocks(role: Role, content: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Map Streamlit message content to Responses API blocks.
    - user  -> input_* types (default to input_text)
    - assistant -> output_text (Responses API requires output_* for assistant history)
    - system -> input_text (but we actually pass system via `instructions`)
    """
    # Assistant history must be output_text
    if role == "assistant":
        text = content if isinstance(content, str) else " ".join(
            [c.get("text", "") for c in content if isinstance(c, dict)]
        )
        return [{"type": "output_text", "text": str(text or "").strip()}]

    # For user/system history, accept multimodal inputs
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    blocks: List[Dict[str, Any]] = []
    for c in content:
        ctype = c.get("type")
        if ctype in ALLOWED_INPUT_TYPES:
            blocks.append(c)
        elif ctype == "text":  # tolerate Message-style
            blocks.append({"type": "input_text", "text": c.get("text", "")})
        # ignore unsupported types for safety
    if not blocks:
        blocks.append({"type": "input_text", "text": ""})
    return blocks

def history_to_responses_input(history: List[ChatMessage]) -> List[Dict[str, Any]]:
    """
    Convert Streamlit history into Responses API input:
    - user/system -> input_* blocks
    - assistant   -> output_text blocks
    """
    inputs: List[Dict[str, Any]] = []
    for m in history:
        role: Role = m.get("role", "user")  # type: ignore
        blocks = normalize_to_content_blocks(role, m.get("content", ""))
        inputs.append({"role": role, "content": blocks})
    return inputs

def build_responses_kwargs(
    model_name: str,
    inputs: List[Dict[str, Any]],
    instructions: str,
    temperature_value: float,
    stream: bool = True,
) -> Dict[str, Any]:
    """
    Build kwargs for client.responses.create, removing temperature for GPT-5 reasoning models.
    Rule: if model starts with 'gpt-5' and NOT a '-chat' variant, omit temperature.
    """
    kwargs: Dict[str, Any] = {
        "model": model_name,
        "input": inputs,
        "instructions": instructions,
        "stream": stream,
    }

    is_gpt5_reasoning = model_name.startswith("gpt-5") and ("chat" not in model_name)
    if not is_gpt5_reasoning:
        kwargs["temperature"] = temperature_value
    # else: omit temperature entirely to avoid 400s

    return kwargs


# ---------- Chat handling ----------
if user_input:
    # 1) Append user's message ONCE to state (no double append later)
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Prepare Inputs for Responses API using ENTIRE history (already includes the latest user turn)
    responses_input = history_to_responses_input(st.session_state.messages)

    # 3) Stream the model reply
    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed_text: List[str] = []

        try:
            kwargs = build_responses_kwargs(
                model_name=model,
                inputs=responses_input,         # <-- no extra user block appended
                instructions=SYSTEM_PROMPT,
                temperature_value=temperature,
                stream=True,
            )

            stream = client.responses.create(**kwargs)

            for event in stream:
                typ = getattr(event, "type", None)
                if typ == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        streamed_text.append(delta)
                        placeholder.markdown("".join(streamed_text))
                elif typ == "response.error":
                    err = getattr(event, "error", {}) or {}
                    msg = err.get("message", "Unknown streaming error.")
                    placeholder.error(msg)

        except Exception as e:
            placeholder.error(f"API error: {e}")

        final_text = "".join(streamed_text).strip()
        if final_text:
            placeholder.markdown(final_text)
            st.session_state.messages.append({"role": "assistant", "content": final_text})
