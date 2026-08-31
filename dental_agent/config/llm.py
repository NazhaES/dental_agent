"""
Single place where the chat model is constructed.

Every agent calls get_llm() instead of instantiating a provider directly, so
switching providers is a one-line change here (or via the LLM_PROVIDER env var).

To go back to grok-4 once xAI credits are restored, either:
  * set LLM_PROVIDER=xai in .env, or
  * change the LLM_PROVIDER default in config/settings.py back to "xai".
"""

from dental_agent.config.settings import (
    LLM_PROVIDER,
    XAI_API_KEY,
    MODEL_NAME,
    GROQ_API_KEY,
    GROQ_MODEL,
    TEMPERATURE,
)


def get_llm():
    """Return the chat model for the active provider."""
    if LLM_PROVIDER == "xai":
        # --- Production path: grok-4 ---
        from langchain_xai import ChatXAI

        if not XAI_API_KEY:
            raise RuntimeError(
                "XAI_API_KEY missing. Set GROK_API_KEY (or XAI_API_KEY) in .env."
            )
        return ChatXAI(
            api_key=XAI_API_KEY,
            model=MODEL_NAME,
            temperature=TEMPERATURE,
        )

    if LLM_PROVIDER == "groq":
        # --- Temporary testing path while xAI credits are exhausted ---
        from langchain_groq import ChatGroq

        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY missing. Add GROQ_API_KEY=... to .env, or set "
                "LLM_PROVIDER=xai to use grok-4."
            )
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=TEMPERATURE,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER {LLM_PROVIDER!r}. Expected 'groq' or 'xai'."
    )
