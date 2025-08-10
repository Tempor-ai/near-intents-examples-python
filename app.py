# app.py
import os
import json
import time
from typing import List, TypedDict, Literal, Dict, Any, Union, Optional
from decimal import Decimal, InvalidOperation

import requests
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

from prompt import SYSTEM_PROMPT, TOKEN_LIST   # <-- comes from prompt.py


st.set_page_config(page_title="Streamlit + OpenAI Responses API", page_icon="💬")

# --- Sidebar (unchanged look) ---
with st.sidebar:
    st.title("⚙️ Settings")
    model_list = ["gpt-4o", "gpt-4o-mini"]   # add gpt-5 variants later; logic below handles temp removal
    model = st.selectbox("Model", model_list, index=0)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# NEAR 1Click API integration
# ----------------------------
BASE_URL = "https://1click.chaindefuser.com"
ONECLICK_JWT = os.getenv("ONECLICK_JWT", "")

DEFAULT_REFUND_TO = os.getenv("REFUND_TO", "0x2527D02599Ba641c19FEa793cD0F167589a0f10D")         # EVM address
DEFAULT_REFUND_TYPE = "ORIGIN_CHAIN"
DEFAULT_RECIPIENT = os.getenv("RECIPIENT", "13QkxhNMrTPxoCkRdYdJ65tFuwXPhL5gLS2Z5Nr6gjRK")       # Solana address
DEFAULT_RECIPIENT_TYPE = "DESTINATION_CHAIN"
DEFAULT_SLIPPAGE_BPS = 100
DEFAULT_DEPOSIT_TYPE = "ORIGIN_CHAIN"
DEFAULT_SWAP_TYPE = "EXACT_INPUT"
DEFAULT_QUOTE_WAIT_MS = 3000
DEFAULT_REFERRAL = os.getenv("REFERRAL", "myapp")

def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {ONECLICK_JWT}" if ONECLICK_JWT else "",
        "Accept": "*/*",
        "Content-Type": "application/json",
    }

# Chain aliases to TOKEN_LIST[i]["blockchain"]
CHAIN_ALIASES = {
    "near":"near",
    "ethereum":"eth","eth":"eth",
    "arbitrum":"arb","arb":"arb",
    "solana":"sol","sol":"sol",
    "base":"base",
    "optimism":"op","op":"op",
    "avalanche":"avax","avax":"avax",
    "polygon":"pol","matic":"pol","pol":"pol",
    "gnosis":"gnosis",
    "bsc":"bsc","binance":"bsc",
    "bitcoin":"btc","btc":"btc",
    "sui":"sui",
    "tron":"tron",
    "cardano":"cardano",
    "bera":"bera",
    "doge":"doge",
    "xrp":"xrp",
    "zec":"zec",
    "ton":"ton",
}

def normalize_chain(name: str) -> str:
    key = (name or "").strip().lower()
    return CHAIN_ALIASES.get(key, key)

def find_token(symbol: str, chain: str) -> Optional[Dict[str, Any]]:
    sym = (symbol or "").strip().lower()
    chn = normalize_chain(chain)
    for t in TOKEN_LIST:
        if t.get("symbol","").lower() == sym and t.get("blockchain","").lower() == chn:
            return t
    return None

def to_atomic_amount(amount_tokens: Union[str, float, int, Decimal], decimals: int) -> str:
    try:
        q = Decimal(str(amount_tokens))
    except (InvalidOperation, ValueError):
        q = Decimal(0)
    factor = Decimal(10) ** Decimal(decimals)
    return str(int((q * factor).to_integral_value(rounding="ROUND_DOWN")))

def request_quote_raw(
    *,
    dry: bool,
    swap_type: str,
    slippage_bps: int,
    origin_asset: str,
    deposit_type: str,
    destination_asset: str,
    amount_atomic: str,
    refund_to: str,
    refund_type: str,
    recipient: str,
    recipient_type: str,
    deadline_iso: Optional[str] = None,
    referral: Optional[str] = None,
    quote_wait_ms: int = DEFAULT_QUOTE_WAIT_MS,
    app_fees: Optional[list] = None,
    virtual_chain_recipient: Optional[str] = None,
    virtual_chain_refund_recipient: Optional[str] = None,
) -> Dict[str, Any]:
    if not deadline_iso:
        # now + 90 minutes ISO8601 Z
        import datetime as dt
        deadline_iso = (dt.datetime.utcnow() + dt.timedelta(minutes=90)).replace(microsecond=0).isoformat() + "Z"

    payload: Dict[str, Any] = {
        "dry": dry,
        "swapType": swap_type,
        "slippageTolerance": slippage_bps,
        "originAsset": origin_asset,
        "depositType": deposit_type,
        "destinationAsset": destination_asset,
        "amount": amount_atomic,
        "refundTo": refund_to,
        "refundType": refund_type,
        "recipient": recipient,
        "recipientType": recipient_type,
        "deadline": deadline_iso,
        "quoteWaitingTimeMs": quote_wait_ms,
    }
    if referral:
        payload["referral"] = referral
    if app_fees:
        payload["appFees"] = app_fees
    if virtual_chain_recipient:
        payload["virtualChainRecipient"] = virtual_chain_recipient
    if virtual_chain_refund_recipient:
        payload["virtualChainRefundRecipient"] = virtual_chain_refund_recipient

    r = requests.post(f"{BASE_URL}/v0/quote", headers=_headers(), json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def tool_get_swap_quote(
    origin_symbol: str,
    origin_chain: str,
    destination_symbol: str,
    destination_chain: str,
    amount_tokens: Union[str, float, int] = "1",
    dry: bool = True,
    swap_type: str = DEFAULT_SWAP_TYPE,
    slippage_bps: int = DEFAULT_SLIPPAGE_BPS,
    deposit_type: str = DEFAULT_DEPOSIT_TYPE,
    recipient: Optional[str] = None,
    recipient_type: str = DEFAULT_RECIPIENT_TYPE,
    refund_to: Optional[str] = None,
    refund_type: str = DEFAULT_REFUND_TYPE,
    referral: Optional[str] = DEFAULT_REFERRAL,
    quote_wait_ms: int = DEFAULT_QUOTE_WAIT_MS,
    app_fees: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Resolve symbols/chains to assetIds using TOKEN_LIST, convert amount to atomic units, then call 1Click /v0/quote."""
    src = find_token(origin_symbol, origin_chain)
    dst = find_token(destination_symbol, destination_chain)
    if not src or not dst:
        return {
            "error": "TOKEN_NOT_FOUND",
            "details": {
                "origin_symbol": origin_symbol, "origin_chain": origin_chain,
                "destination_symbol": destination_symbol, "destination_chain": destination_chain
            }
        }

    amt_atomic = to_atomic_amount(amount_tokens, int(dst["decimals"] if swap_type != "EXACT_INPUT" else src["decimals"]))
    # For EXACT_INPUT we convert using origin decimals; for EXACT_OUTPUT you'd convert destination amount requested.
    if swap_type == "EXACT_OUTPUT":
        amt_atomic = to_atomic_amount(amount_tokens, int(dst["decimals"]))

    q = request_quote_raw(
        dry=dry,
        swap_type=swap_type,
        slippage_bps=slippage_bps,
        origin_asset=src["assetId"],
        deposit_type=deposit_type,
        destination_asset=dst["assetId"],
        amount_atomic=amt_atomic,
        refund_to=refund_to or DEFAULT_REFUND_TO,
        refund_type=refund_type,
        recipient=recipient or DEFAULT_RECIPIENT,
        recipient_type=recipient_type,
        referral=referral,
        quote_wait_ms=quote_wait_ms,
        app_fees=app_fees,
    )
    return {
        "origin": {"symbol": src["symbol"], "chain": src["blockchain"], "assetId": src["assetId"], "decimals": src["decimals"]},
        "destination": {"symbol": dst["symbol"], "chain": dst["blockchain"], "assetId": dst["assetId"], "decimals": dst["decimals"]},
        "quote": q
    }

# ----------------------------
# Responses API tool schema
# ----------------------------
TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "name": "get_swap_quote",
        "description": (
            "Get a cross-chain swap quote via NEAR 1Click. "
            "Use when the user asks for a quote/route/price to move an asset from an origin chain to a destination chain. "
            "Resolve asset symbols and chains using the provided TOKEN_LIST only (case-insensitive). "
            "Prefer EXACT_INPUT with amount in tokens unless the user specifies exact output."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "origin_symbol": {"type": "string", "description": "Symbol on origin chain, e.g., 'USDC'"},
                "origin_chain": {"type": "string", "description": "Chain name or alias, e.g., 'arbitrum'|'arb'|'solana'|'sol'|'eth'|'ethereum'|'near'"},
                "destination_symbol": {"type": "string", "description": "Symbol on destination chain, e.g., 'USDC'"},
                "destination_chain": {"type": "string", "description": "Chain name or alias, e.g., 'sol'|'solana'|'arb'|'arbitrum'"},
                "amount_tokens": {"type": ["string", "number"], "description": "Human amount in tokens, e.g., '1'"},
                "dry": {"type": "boolean", "default": True, "description": "True to preview without creating a deposit address"},
                "swap_type": {"type": "string", "enum": ["EXACT_INPUT", "EXACT_OUTPUT", "FLEX_INPUT"], "default": "EXACT_INPUT"},
                "slippage_bps": {"type": "integer", "default": 100, "description": "Slippage tolerance in basis points"},
                "deposit_type": {"type": "string", "enum": ["ORIGIN_CHAIN", "INTENTS"], "default": "ORIGIN_CHAIN"},
                "recipient": {"type": "string", "description": "Destination recipient address. If omitted, a default is used."},
                "recipient_type": {"type": "string", "enum": ["DESTINATION_CHAIN", "INTENTS"], "default": "DESTINATION_CHAIN"},
                "refund_to": {"type": "string", "description": "Refund address. If omitted, a default EVM address is used."},
                "refund_type": {"type": "string", "enum": ["ORIGIN_CHAIN", "INTENTS"], "default": "ORIGIN_CHAIN"},
                "referral": {"type": "string", "description": "Optional referral/app tag"},
                "quote_wait_ms": {"type": "integer", "default": 3000},
                "app_fees": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"recipient": {"type": "string"}, "fee": {"type": "integer"}}
                    }
                }
            },
            "required": ["origin_symbol", "origin_chain", "destination_symbol", "destination_chain", "amount_tokens"]
        }
    }
]

# ----------------------------
# Streamlit chat state & render
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role":"user"|"assistant","content": str|blocks}]

st.title("InstaSwap Python Assistant")
st.markdown("Powered by NEAR 1Click Intents")

# Render history (unchanged look)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        text = m["content"]
        if isinstance(text, str):
            st.markdown(text)
        else:
            txt_parts = [c.get("text","") for c in text if isinstance(c, dict) and c.get("type") in ("input_text","summary_text")]
            st.markdown("\n\n".join([t for t in txt_parts if t]))

# ---------- Helpers: content typing ----------
Role = Literal["user","assistant","system"]

class ChatMessage(TypedDict):
    role: Role
    content: Union[str, List[Dict[str, Any]]]

ALLOWED_INPUT_TYPES = {"input_text","input_image","input_file","computer_screenshot","summary_text"}

def normalize_to_content_blocks(role: Role, content: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if role == "assistant":
        text = content if isinstance(content, str) else " ".join([c.get("text","") for c in content if isinstance(c, dict)])
        return [{"type": "output_text", "text": str(text or "").strip()}]
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    blocks: List[Dict[str, Any]] = []
    for c in content:
        ctype = c.get("type")
        if ctype in ALLOWED_INPUT_TYPES:
            blocks.append(c)
        elif ctype == "text":
            blocks.append({"type":"input_text","text": c.get("text","")})
    if not blocks:
        blocks.append({"type":"input_text","text":""})
    return blocks

def history_to_responses_input(history: List[ChatMessage]) -> List[Dict[str, Any]]:
    inputs: List[Dict[str, Any]] = []
    for m in history:
        role: Role = m.get("role","user")  # type: ignore
        inputs.append({"role": role, "content": normalize_to_content_blocks(role, m.get("content",""))})
    return inputs

def build_kwargs(model_name: str, inputs: List[Dict[str, Any]], instructions: str, temperature_value: float, stream: bool=True) -> Dict[str, Any]:
    kw: Dict[str, Any] = {"model": model_name, "input": inputs, "instructions": instructions, "tools": TOOLS, "tool_choice": "auto", "stream": stream}
    is_gpt5_reasoning = model_name.startswith("gpt-5") and ("chat" not in model_name)
    if not is_gpt5_reasoning:
        kw["temperature"] = temperature_value
    return kw

# ---------- Tool router (exec + submit outputs) ----------
def handle_tool_calls_and_stream(response_id: str, pending_tool_calls: List[Dict[str, Any]], placeholder) -> str:
    """Execute local tools, then submit outputs and stream the model's final answer."""
    outputs = []
    for tc in pending_tool_calls:
        name = tc.get("name")
        args = tc.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if name == "get_swap_quote":
            try:
                result = tool_get_swap_quote(**args)
                outputs.append({"tool_call_id": tc["id"], "output": json.dumps(result)})
            except Exception as e:
                outputs.append({"tool_call_id": tc["id"], "output": json.dumps({"error":"TOOL_EXEC_ERROR","message": str(e)})})
        else:
            outputs.append({"tool_call_id": tc["id"], "output": json.dumps({"error":"UNKNOWN_TOOL"})})

    streamed_text: List[str] = []
    stream2 = client.responses.submit_tool_outputs(
        response_id=response_id,
        tool_outputs=outputs,
        stream=True
    )
    for event in stream2:
        typ = getattr(event, "type", None)
        if typ == "response.output_text.delta":
            delta = getattr(event, "delta", "")
            if delta:
                streamed_text.append(delta)
                placeholder.markdown("".join(streamed_text))
        elif typ == "response.error":
            err = getattr(event, "error", {}) or {}
            placeholder.error(err.get("message", "Unknown streaming error."))

    final_text = "".join(streamed_text).strip()
    return final_text

# ---------- Chat loop ----------
user_input = st.chat_input("Type your message…")
if user_input:
    # 1) Save user turn (no double-append later)
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Build inputs from full history
    inputs = history_to_responses_input(st.session_state.messages)

    # 3) Create response (stream first; intercept tool calls; then submit outputs & stream)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed_text: List[str] = []
        pending_tool_calls: List[Dict[str, Any]] = []
        response_id: Optional[str] = None

        try:
            stream = client.responses.create(**build_kwargs(model, inputs, SYSTEM_PROMPT, temperature, stream=True))

            for event in stream:
                et = getattr(event, "type", None)

                # capture response id
                if et == "response.created":
                    # best-effort to fetch id from event
                    rid = getattr(event, "response", None)
                    response_id = getattr(rid, "id", None) if rid else getattr(event, "id", None)

                # normal text streaming
                if et == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        streamed_text.append(delta)
                        placeholder.markdown("".join(streamed_text))

                # tool call streaming (collect; we’ll execute after first stream completes)
                if et == "response.tool_call.delta":
                    # accumulate partial arguments by tool_call_id
                    tc = getattr(event, "tool_call", {}) or {}
                    if tc:
                        # merge by id
                        existing = next((t for t in pending_tool_calls if t.get("id")==tc.get("id")), None)
                        if existing:
                            # append partial args text
                            existing["arguments"] = (existing.get("arguments","") or "") + (tc.get("arguments_delta","") or "")
                        else:
                            pending_tool_calls.append({
                                "id": tc.get("id"),
                                "name": tc.get("name"),
                                "arguments": tc.get("arguments_delta","") or ""
                            })

                if et == "response.tool_call.completed":
                    tc = getattr(event, "tool_call", {}) or {}
                    if tc:
                        # ensure we have the final args for this tool_call id
                        for i, p in enumerate(pending_tool_calls):
                            if p.get("id") == tc.get("id"):
                                pending_tool_calls[i]["arguments"] = tc.get("arguments","") or p.get("arguments","")
                                pending_tool_calls[i]["name"] = tc.get("name", p.get("name"))

                if et == "response.error":
                    err = getattr(event, "error", {}) or {}
                    placeholder.error(err.get("message", "Unknown streaming error."))

            # If we received tool calls, execute them and stream the final answer
            if pending_tool_calls and response_id:
                final_text = handle_tool_calls_and_stream(response_id, pending_tool_calls, placeholder)
                if final_text:
                    placeholder.markdown(final_text)
                    st.session_state.messages.append({"role": "assistant", "content": final_text})
            else:
                # no tools used; finalize plain streamed text
                final_text = "".join(streamed_text).strip()
                if final_text:
                    placeholder.markdown(final_text)
                    st.session_state.messages.append({"role":"assistant","content": final_text})

        except Exception as e:
            placeholder.error(f"API error: {e}")
