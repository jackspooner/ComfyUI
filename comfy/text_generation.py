"""Serializable text conversations and exact token-window selection."""
import json


def read_history(history, prompt, system_prompt=""):
    value = json.loads(history or '{"version":1,"messages":[]}')
    if not isinstance(value, dict) or set(value) != {"version", "messages"} or type(value["version"]) is not int or value["version"] != 1 or not isinstance(value["messages"], list):
        raise ValueError("History must contain version 1 and a messages array.")
    messages = []
    for index, message in enumerate(value["messages"]):
        if not isinstance(message, dict) or set(message) != {"role", "content"} or not isinstance(message["content"], str):
            raise ValueError("History messages require only role and string content.")
        role = message["role"]
        if role not in ("system", "user", "assistant") or (role == "system" and index != 0):
            raise ValueError("History accepts user and assistant text, with one optional leading system message.")
        messages.append(dict(message))
    if system_prompt:
        if messages and messages[0]["role"] == "system":
            messages = messages[1:]
        messages.insert(0, {"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def format_qwen_chat(messages, thinking):
    text = "".join(f'<|im_start|>{message["role"]}\n{message["content"]}<|im_end|>\n' for message in messages)
    return text + "<|im_start|>assistant\n" + ("" if thinking else "<think>\n\n</think>\n\n")


def select_context(messages, tokenize, model_limit, context_budget, max_length, history_policy):
    if type(context_budget) is not int or context_budget < 0 or context_budget > model_limit:
        raise ValueError(f"context_budget must be 0 or between 1 and {model_limit} tokens.")
    if type(max_length) is not int or max_length < 0:
        raise ValueError("max_length must be a nonnegative integer.")
    if history_policy not in ("error", "drop_oldest_turns"):
        raise ValueError("history_policy must be error or drop_oldest_turns.")
    limit = context_budget or model_limit
    active = list(messages)
    excluded = []
    offset = int(active[0]["role"] == "system")
    removed = 0
    while True:
        tokens = tokenize(active)
        if len(tokens) != 1:
            raise ValueError("Conversation token accounting requires one generative tokenizer stream.")
        rows = next(iter(tokens.values()))
        if any(type(item[0]) is not int for row in rows for item in row):
            raise ValueError("Text history does not support media embeddings.")
        count = sum(len(row) for row in rows)
        if count + (max_length or 1) <= limit:
            return tokens, {
                "version": 1, "model_context_limit": model_limit, "context_limit": limit,
                "input_tokens": count, "output_allowance": max_length or limit - count,
                "excluded_message_ranges": excluded,
            }
        if history_policy == "drop_oldest_turns" and len(active) > offset + 2 and active[offset]["role"] == "user" and active[offset + 1]["role"] == "assistant":
            # Half-open ranges refer to the full, normalized input transcript.
            excluded.append([offset + removed, offset + removed + 2])
            del active[offset:offset + 2]
            removed += 2
            continue
        raise ValueError(f"Context overflow: input uses {count} tokens and needs {max_length or 1} output tokens within {limit}. Shorten input or output, or explicitly select drop_oldest_turns for complete prior turns.")


def write_history(messages):
    return json.dumps({"version": 1, "messages": messages}, ensure_ascii=False)
