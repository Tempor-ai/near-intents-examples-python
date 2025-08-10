# app.py
import os
import streamlit as st
from openai import OpenAI
from prompt import SYSTEM_PROMPT
from dotenv import load_dotenv
load_dotenv()


st.set_page_config(page_title="Streamlit + OpenAI Responses API", page_icon="💬")

# --- Sidebar config ---
with st.sidebar:
    st.title("⚙️ Settings")
    # model_list = ["gpt-5", "gpt-5-mini"]
    model_list = ["gpt-4o", "gpt-4o-mini"]
    model = st.selectbox("Model", model_list, index=0)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Session state for chat history (Streamlit-native chat UI) ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # each item: {"role": "user"|"assistant", "content": "..."}

st.title("NEAR Intents Swap Quote Assistant")

# Render the history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input box
user_input = st.chat_input("Type your message…")

def history_to_responses_input(history: list[dict[str, str]]) -> list[dict]:
    """
    Convert Streamlit's simple message list into the Responses API 'input' format:
    [{ role, content: [{type: 'input_text', text: ...}] }, ...]
    """
    formatted = []
    # include a 'system'/instructions layer via the dedicated 'instructions' parameter below
    for m in history:
        formatted.append({
            "role": m["role"],
            "content": [{"type": "input_text", "text": m["content"]}],
        })
    return formatted

if user_input:
    # 1) Echo user message locally
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Prepare Inputs for Responses API
    responses_input = history_to_responses_input(st.session_state.messages)

    # 3) Stream the model reply
    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed_text = []

        try:
            stream = client.responses.create(
                model=model,
                input=responses_input + [{
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_input}],
                }],
                instructions=SYSTEM_PROMPT,   # system-like guidance
                temperature=temperature,
                stream=True,                  # <-- stream events
            )

            for event in stream:
                # Only append text deltas; ignore other event types (created, completed, etc.)
                # See Responses API streaming event types like `response.output_text.delta`.
                typ = getattr(event, "type", None)
                if typ == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        streamed_text.append(delta)
                        placeholder.markdown("".join(streamed_text))
                elif typ == "response.error":
                    # Surface API-side errors gracefully
                    err = getattr(event, "error", {}) or {}
                    msg = err.get("message", "Unknown streaming error.")
                    placeholder.error(msg)

        except Exception as e:
            placeholder.error(f"API error: {e}")

        final_text = "".join(streamed_text).strip()
        if final_text:
            placeholder.markdown(final_text)
            st.session_state.messages.append({"role": "assistant", "content": final_text})
