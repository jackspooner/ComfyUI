"""Structured chat assembly; image slots never come from literal user text."""


def chat_segments(messages, role_start, turn_end, assistant_start, image_start, image_end, prefix=""):
    text = prefix
    for message in messages:
        role = message["role"]
        text += role_start.format(role=role)
        content = message["content"]
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
        for part in parts:
            if part["type"] == "text":
                text += part["text"]
            else:
                yield text + image_start
                yield {"type": "image", "data": part["data"]}
                text = image_end
        text += turn_end
    yield text + assistant_start


def encode_chat(segments, encode, prepare_image=None, start_tokens=()):
    ids = list(start_tokens)
    for segment in segments:
        if isinstance(segment, str):
            ids.extend(encode(segment))
        else:
            ids.append(prepare_image(segment) if prepare_image else segment)
    return [[(token, 1.0) for token in ids]]


def qwen_chat(messages, tokenizer, thinking=None):
    assistant = "<|im_start|>assistant\n"
    if thinking is False:
        assistant += "<think>\n\n</think>\n\n"
    segments = chat_segments(messages, "<|im_start|>{role}\n", "<|im_end|>\n", assistant,
                             "<|vision_start|>", "<|vision_end|>")
    return encode_chat(segments, lambda text: tokenizer.encode(text, add_special_tokens=False))


def count_chat_tokens(tokens, image_length=None):
    if isinstance(tokens, dict):
        if len(tokens) != 1:
            raise ValueError("Conversation token accounting requires one generative tokenizer stream.")
        tokens = next(iter(tokens.values()))
    count = 0
    for row in tokens:
        for token, *_ in row:
            if isinstance(token, int):
                count += 1
            elif isinstance(token, dict) and token.get("type") == "image" and image_length is not None:
                length = image_length(token)
                token["expected_length"] = length
                count += length
            else:
                raise ValueError("This model does not support images in conversation history.")
    return count
