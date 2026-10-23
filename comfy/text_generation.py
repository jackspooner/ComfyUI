"""Serializable text conversations and exact token-window selection."""
import json


class ContextOverflowError(ValueError):
    pass


def read_content(content, images=False):
    if isinstance(content, str):
        return content
    if not images or not isinstance(content, list):
        raise ValueError("History content must be text or version 2 content parts.")
    parts = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("History content parts must be objects.")
        if part.get("type") == "text" and set(part) == {"type", "text"} and isinstance(part["text"], str):
            parts.append(dict(part))
        elif part.get("type") == "image" and set(part) == {"type", "image_id"} and isinstance(part["image_id"], str):
            parts.append(dict(part))
        elif part.get("type") == "image" and set(part) == {"type", "file"} and isinstance(part["file"], dict) and set(part["file"]) == {"filename", "subfolder"} and all(isinstance(v, str) for v in part["file"].values()):
            parts.append({"type": "image", "file": dict(part["file"])})
        else:
            raise ValueError("Image history requires text, image_id or a local file with filename and subfolder.")
    return parts


def read_history(history, prompt, system_prompt="", prompt_content=""):
    value = json.loads(history or '{"version":1,"messages":[]}')
    if not isinstance(value, dict) or set(value) != {"version", "messages"} or type(value["version"]) is not int or value["version"] not in (1, 2) or not isinstance(value["messages"], list):
        raise ValueError("History must contain version 1 or 2 and a messages array.")
    messages = []
    for index, message in enumerate(value["messages"]):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("History messages require only role and content.")
        role = message["role"]
        if role not in ("system", "user", "assistant") or (role == "system" and index != 0):
            raise ValueError("History accepts user and assistant text, with one optional leading system message.")
        messages.append({"role": role, "content": read_content(message["content"], value["version"] == 2 and role != "system")})
    if system_prompt:
        if messages and messages[0]["role"] == "system":
            messages = messages[1:]
        messages.insert(0, {"role": "system", "content": system_prompt})
    content = read_content(json.loads(prompt_content), True) if prompt_content else prompt
    if prompt_content and not isinstance(content, list):
        raise ValueError("prompt_content must be a JSON content-parts array.")
    messages.append({"role": "user", "content": content})
    return messages


def format_qwen_chat(messages, thinking):
    if any(not isinstance(message["content"], str) for message in messages):
        raise ValueError("This text-only adapter does not support image history.")
    text = "".join(f'<|im_start|>{message["role"]}\n{message["content"]}<|im_end|>\n' for message in messages)
    return text + "<|im_start|>assistant\n" + ("" if thinking else "<think>\n\n</think>\n\n")


def count_text_tokens(tokens):
    if isinstance(tokens, dict):
        if len(tokens) != 1:
            raise ValueError("Conversation token accounting requires one generative tokenizer stream.")
        tokens = next(iter(tokens.values()))
    rows = tokens
    if any(type(item[0]) is not int for row in rows for item in row):
        raise ValueError("Text history does not support media embeddings.")
    return sum(len(row) for row in rows)


def select_context(messages, tokenize, model_limit, context_budget, max_length, history_policy, count_tokens=count_text_tokens, prepare_tokens=None):
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
        media_info = {}
        overflow = None
        try:
            if prepare_tokens is not None:
                tokens, media_info = prepare_tokens(tokens)
            count = count_tokens(tokens)
        except ContextOverflowError as error:
            overflow = error
            count = limit + 1
        if count + (max_length or 1) <= limit:
            info = {
                "version": 1, "model_context_limit": model_limit, "context_limit": limit,
                "input_tokens": count, "output_allowance": max_length or limit - count,
                **media_info,
                "excluded_message_ranges": excluded,
            }
            return tokens, info
        if history_policy == "drop_oldest_turns" and len(active) > offset + 2 and active[offset]["role"] == "user" and active[offset + 1]["role"] == "assistant":
            # Half-open ranges refer to the full, normalized input transcript.
            excluded.append([offset + removed, offset + removed + 2])
            del active[offset:offset + 2]
            removed += 2
            continue
        if overflow is not None:
            raise overflow
        raise ContextOverflowError(f"Context overflow: input uses {count} tokens and needs {max_length or 1} output tokens within {limit}. Shorten input or output, or explicitly select drop_oldest_turns for complete prior turns.")


def write_history(messages):
    normalized = []
    has_images = False
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            if all(part["type"] == "text" for part in content):
                content = "".join(part["text"] for part in content)
            else:
                has_images = True
        normalized.append({"role": message["role"], "content": content})
    return json.dumps({"version": 2 if has_images else 1, "messages": normalized}, ensure_ascii=False)
