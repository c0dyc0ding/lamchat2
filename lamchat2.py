#!/usr/bin/env python3
"""
Multi-Model Streaming Chat Client with LangChain and SQLite Memory

This client lets you interactively chat (via text) with multiple Ollama models.
• You provide a default model via "--model" and may optionally supply two extra models (--model_a and --model_b).
• At any turn, you may prefix your message with "@A:" or "@B:" (for example) to query the respective model.
• If no prefix is provided, the default model is used.
• All messages (user input and AI responses) are stored in a shared SQLite database.
• The conversation is streamed (tokens printed in real time), and an optional logfile records the transcript.
• (Use --debug to print the raw JSON tokens from the server for troubleshooting.)

Before running:
  - Install dependencies:
        pip install -U httpx langchain pydantic
  - Set up and start your local Ollama instance (see: https://github.com/ollama/ollama)
  - Pull your desired models (e.g., via "ollama pull <model_name>")

Usage examples:
    python3 multi_model_streaming_chat.py --model "llama3"
    python3 multi_model_streaming_chat.py --model "default-model" --model_a "cens" --model_b "llama3" --logfile "conversation.log" --debug
"""

import argparse
import asyncio
import json
import sqlite3
import httpx
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, ConfigDict

# Import LangChain classes and schema.
from langchain.llms.base import LLM
from langchain.schema import BaseChatMessageHistory, HumanMessage, AIMessage


###############################################################################
# Utility: Safe JSON Parsing
###############################################################################
def safe_json_loads(text: str) -> Any:
    """
    Safely parse a string into a JSON object.
    If extra data exists, iterates line by line until valid JSON is found.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        raise e


###############################################################################
# SQLite-Based Shared Chat Memory (subclassing BaseChatMessageHistory)
###############################################################################
class SQLiteChatMemory(BaseChatMessageHistory):
    """
    A persistent conversation memory backed by SQLite.
    Every message (from Human or AI) is stored.
    This shared memory is used by all models so that each model sees the full history.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS messages ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "role TEXT NOT NULL, "
            "content TEXT NOT NULL, "
            "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        self.conn.commit()

    def add_message(self, message: Any) -> None:
        if isinstance(message, HumanMessage):
            role = "Human"
        elif isinstance(message, AIMessage):
            role = "AI"
        else:
            role = "Other"
        self.conn.execute(
            "INSERT INTO messages (role, content) VALUES (?, ?)",
            (role, message.content),
        )
        self.conn.commit()

    def add_user_message(self, content: str) -> None:
        self.add_message(HumanMessage(content=content))

    def add_ai_message(self, content: str) -> None:
        self.add_message(AIMessage(content=content))

    @property
    def messages(self) -> List[Any]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT role, content FROM messages ORDER BY id ASC")
        rows = cursor.fetchall()
        out = []
        for role, content in rows:
            if role == "Human":
                out.append(HumanMessage(content=content))
            else:
                out.append(AIMessage(content=content))
        return out

    def clear(self) -> None:
        self.conn.execute("DELETE FROM messages")
        self.conn.commit()


###############################################################################
# Simplified Ollama LLM Integration with Streaming
###############################################################################
class OllamaLLM(LLM, BaseModel):
    """
    A streamlined LLM integration for Ollama with streaming support.
    
    When using a chat endpoint, the payload includes the conversation history 
    (a list of messages as dicts) plus the current user prompt.
    
    Responses are streamed token by token and printed in real time. An optional
    debug flag prints each raw JSON token from the server.
    
    Generation behavior is governed by your Ollama modelfile/server defaults.
    """
    model: str
    base_url: Optional[str] = None  # Defaults to "http://localhost:11434" if not provided.
    completion_endpoint: Optional[str] = "/api/chat"
    debug: bool = False

    model_config = ConfigDict(extra="forbid")
    
    @property
    def _llm_type(self) -> str:
        return "ollama"
    
    def _build_endpoint(self) -> str:
        base_url = self.base_url or "http://localhost:11434"
        return f"{base_url.rstrip('/')}/{self.completion_endpoint.lstrip('/')}"
    
    def _build_payload(self, prompt: str, history: Optional[List[dict]] = None, stop: Optional[List[str]] = None) -> dict[str, Any]:
        # For a chat endpoint, include the history (if any) plus the new user message.
        if "chat" in self.completion_endpoint.lower():
            messages = history if history is not None else []
            messages.append({"role": "user", "content": prompt})
            return {"model": self.model, "messages": messages}
        else:
            return {"model": self.model, "prompt": prompt}
    
    async def stream_call(self, prompt: str, history: Optional[List[dict]] = None, stop: Optional[List[str]] = None) -> str:
        """
        Makes an asynchronous streaming call to the Ollama server.
        Sends the conversation history (if any) along with the current prompt.
        Tokens are printed as they arrive and concatenated into a full answer.
        """
        endpoint = self._build_endpoint()
        payload = self._build_payload(prompt, history, stop)
        full_text = ""
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", endpoint, json=payload) as response:
                if response.status_code != 200:
                    raise Exception(f"Request failed ({response.status_code}): {response.text}")
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = safe_json_loads(line)
                        if self.debug:
                            print("\nDEBUG token JSON:", json.dumps(data, indent=2))
                        token = ""
                        if "message" in data and isinstance(data["message"], dict):
                            token = data["message"].get("content", "")
                        full_text += token
                        print(token, end="", flush=True)
                        if data.get("done", False):
                            break
                    except Exception:
                        continue
        return full_text

    # We override the synchronous _call to indicate streaming is required.
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        raise NotImplementedError("This LLM only supports streaming calls; use stream_call().")


###############################################################################
# Asynchronous input helper
###############################################################################
async def async_input(prompt: str = "") -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)


###############################################################################
# Asynchronous Multi-Model Streaming Chat Loop
###############################################################################
async def async_multi_model_chat(models: Dict[str, OllamaLLM], sqlite_memory: SQLiteChatMemory, logfile: Optional[str] = None):
    """
    An asynchronous streaming chat loop that supports multiple models.
    
    The user can prefix their message with a model key (e.g. "@A:" or "@B:").
    If no prefix is provided, the default model (key "default") is used.
    
    All conversation history is stored in SQLite and shared among models.
    Responses are streamed token-by-token, and each response is annotated
    with the model key that produced it.
    
    Supported commands:
       /help   -> Show help text.
       /clear  -> Clear conversation memory.
       exit/quit -> End the chat.
    """
    log_fp = None
    if logfile:
        try:
            log_fp = open(logfile, "a")
            print(f"[Logging conversation to '{logfile}']")
        except Exception as e:
            print(f"[Warning: Could not open logfile '{logfile}': {e}]")
    
    print("\nWelcome to the Multi-Model Streaming Chat Client!")
    print("Type your messages below.")
    print("Use '@<key>:' at the start of your message to choose a model (e.g., '@A: Tell a joke').")
    print("If no prefix is given, the default model is used.")
    print("Commands: /help, /clear, exit, quit\n")
    
    while True:
        try:
            user_input = await async_input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break
        
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if user_input == "/help":
            print("Commands: /help (show commands), /clear (clear memory), exit/quit (exit chat)")
            continue
        if user_input == "/clear":
            sqlite_memory.clear()
            print("[Conversation memory cleared]")
            continue
        
        # Determine if a model prefix is provided.
        selected_key = "default"
        query = user_input
        if user_input.startswith("@") and ":" in user_input:
            # Expected format: "@X: message" where X is a key such as A or B.
            split_index = user_input.find(":")
            key = user_input[1:split_index].strip().upper()  # e.g., "A"
            if key in models:
                selected_key = key
                query = user_input[split_index+1:].strip()
            else:
                print(f"[No model mapped for key '{key}', using default]")
        
        # Annotate user message with the chosen model key.
        annotated_user = f"(to {selected_key}) {query}"
        if log_fp:
            log_fp.write("You: " + annotated_user + "\n")
        sqlite_memory.add_user_message(annotated_user)
        
        # Retrieve conversation history from SQLite and convert to list of dicts.
        history = []
        for msg in sqlite_memory.messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        
        # Use the selected model.
        selected_model = models.get(selected_key, models["default"])
        print(f"{selected_key} Model is responding:")
        print("Response: ", end="", flush=True)
        try:
            response_text = await selected_model.stream_call(query, history=history)
            print()  # New line after streaming.
            # Annotate the AI message with the model key.
            annotated_response = f"[{selected_key}] {response_text}"
            sqlite_memory.add_ai_message(annotated_response)
            if log_fp:
                log_fp.write("Model " + selected_key + ": " + annotated_response + "\n")
                log_fp.flush()
        except Exception as e:
            print(f"\n[Error during generation: {e}]")
    
    if log_fp:
        log_fp.close()


###############################################################################
# Main: Parse arguments and launch the asynchronous chat client.
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Multi-Model Streaming Chat Client using Ollama LLM, LangChain, and SQLite for persistent conversation memory."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the default Ollama model (e.g., 'llama3')."
    )
    parser.add_argument(
        "--model_a",
        type=str,
        default=None,
        help="Name of model A (optional). Use prefix '@A:' to query this model."
    )
    parser.add_argument(
        "--model_b",
        type=str,
        default=None,
        help="Name of model B (optional). Use prefix '@B:' to query this model."
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Base URL for the Ollama server (default: http://localhost:11434)."
    )
    parser.add_argument(
        "--completion_endpoint",
        type=str,
        default="/api/chat",
        help="Endpoint for completions (default: '/api/chat')."
    )
    parser.add_argument(
        "--sqlite_db",
        type=str,
        default="chat_history.db",
        help="Path to the SQLite database for persistent conversation memory (default: chat_history.db)."
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default=None,
        help="Path to a file where the conversation will be logged (optional)."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full raw JSON tokens from the server for debugging."
    )
    args = parser.parse_args()

    # Create the default model instance.
    model_default = OllamaLLM(
        model=args.model,
        base_url=args.base_url,
        completion_endpoint=args.completion_endpoint,
        debug=args.debug,
    )

    # Build a dictionary of models. "default" is always present.
    models: Dict[str, OllamaLLM] = {"default": model_default}
    if args.model_a:
        models["A"] = OllamaLLM(
            model=args.model_a,
            base_url=args.base_url,
            completion_endpoint=args.completion_endpoint,
            debug=args.debug,
        )
    if args.model_b:
        models["B"] = OllamaLLM(
            model=args.model_b,
            base_url=args.base_url,
            completion_endpoint=args.completion_endpoint,
            debug=args.debug,
        )
    
    # Launch shared SQLite chat memory.
    sqlite_memory = SQLiteChatMemory(args.sqlite_db)
    
    # Run the asynchronous streaming chat loop.
    asyncio.run(async_multi_model_chat(models, sqlite_memory, logfile=args.logfile))


if __name__ == "__main__":
    main()
