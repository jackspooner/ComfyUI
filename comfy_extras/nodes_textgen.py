from comfy_api.latest import ComfyExtension, io
from typing_extensions import override
import json
import logging
from comfy.text_generation import read_history, select_context, write_history, count_text_tokens
import comfy.model_management
from comfy_extras.text_history_images import resolve_images, publish_images, fingerprint

class TextGenerate(io.ComfyNode):
    history_output = True

    @classmethod
    def define_schema(cls):
        # Define dynamic combo options for sampling mode
        sampling_options = [
            io.DynamicCombo.Option(
                key="on",
                inputs=[
                    io.Float.Input("temperature", default=0.7, min=0.01, max=2.0, step=0.000001),
                    io.Int.Input("top_k", default=64, min=0, max=1000),
                    io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("min_p", default=0.05, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("repetition_penalty", default=1.05, min=0.0, max=5.0, step=0.01),
                    io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                    io.Float.Input("presence_penalty", optional=True, default=0.0, min=0.0, max=5.0, step=0.01),
                ]
            ),
            io.DynamicCombo.Option(
                key="off",
                inputs=[]
            ),
        ]

        return io.Schema(
            node_id="TextGenerate",
            display_name="Generate Text",
            category="text",
            search_aliases=["LLM", "gemma"],
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Image.Input("image", optional=True),
                io.Image.Input("video", optional=True, tooltip="Video frames as an IMAGE batch. Set source_fps to its playback rate and fps to the number of frames to analyse per second."),
                io.Audio.Input("audio", optional=True),
                io.Int.Input("max_length", default=512, min=0, max=1048576, tooltip="Maximum new tokens, including thinking. 0 uses remaining context after prompt/media (remaining affordable capacity with auto_resize). Use a positive value to reserve useful output before media sizing. Generation stops at the model's end token."),
                io.DynamicCombo.Input("sampling_mode", options=sampling_options, display_name="Sampling Mode"),
                io.Boolean.Input("thinking", optional=True, default=False, tooltip="Operate in thinking mode if the model supports it."),
                io.Boolean.Input("use_default_template", optional=True, default=True, tooltip="Use the built in system prompt/template if the model has one.", advanced=True),
                io.Combo.Input("mtp", options=["auto", "off", "2", "3", "4", "5"], default="auto", optional=True, tooltip="Speculative decoding with the checkpoint's multi-token-prediction head. No effect without MTP weights. auto adapts the draft depth; 2-5 pins it. Sampled output stays correctly distributed but differs from non-MTP output for the same seed."),
                io.String.Input("system_prompt", force_input=True, optional=True, tooltip="Replaces the system prompt in the model's chat template. Ignored when the default template is not used."),
                io.String.Input("history", force_input=True, optional=True, tooltip='Version 1 text or version 2 image history JSON. Start with {"version":1,"messages":[]}. Empty keeps the existing single prompt path.'),
                io.String.Input("prompt_content", force_input=True, optional=True, tooltip="Ordered JSON text/image parts for the current user message. Requires empty prompt and no IMAGE input. Local input files or saved image IDs only."),
                io.Int.Input("context_budget", default=0, min=0, max=1048576, optional=True, tooltip="Total input and output token ceiling for text/images/video. 0 uses the model limit. With auto_resize, available device memory can lower the usable window further. Audio requires auto_resize and exact model token accounting."),
                io.Combo.Input("history_policy", options=["error", "drop_oldest_turns"], default="error", optional=True, tooltip="Fail on overflow, or explicitly exclude oldest complete turns from this generation. Returned history stays complete. Does not apply to audio/video inputs."),
                io.Float.Input("fps", default=1.0, min=0.01, max=1000.0, step=0.01, optional=True, tooltip="Frames to analyse per source second: 4 selects one frame every 0.25 seconds. Never duplicates source frames."),
                io.Float.Input("source_fps", default=24.0, min=0.01, max=1000.0, step=0.01, optional=True, tooltip="Actual playback rate of the supplied IMAGE batch. IMAGE batches do not retain video FPS metadata."),
                io.Boolean.Input("auto_resize", default=False, optional=True, tooltip="Maximize supported image/frame sizes within available device memory and the token ceiling, preserving max_length. Uses model-aware memory estimates; native model limits still apply."),
            ],
            outputs=[
                io.String.Output(display_name="generated_text"),
                io.String.Output(display_name="thinking"),
                io.String.Output(display_name="history"),
                io.String.Output(display_name="context_info"),
                io.String.Output(display_name="raw_text"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, history="", prompt_content="", **kwargs):
        if history is None or prompt_content is None:
            return float("nan")  # Linked histories are unavailable during the constants-only fingerprint pass.
        return fingerprint(read_history(history, "", prompt_content=prompt_content))

    @classmethod
    def execute(cls, clip, prompt, max_length, sampling_mode, image=None, thinking=False, use_default_template=True, video=None, audio=None, mtp="auto", system_prompt="", history="", context_budget=0, history_policy="error", prompt_content="", fps=1.0, source_fps=24.0, auto_resize=False) -> io.NodeOutput:

        mtp = False if mtp == "off" else (True if mtp == "auto" else int(mtp))

        messages = None
        context_info = None
        stored_messages = None
        pending_images = {}
        video_info = {}
        unplanned_tokens = None

        def prepare(value):
            nonlocal unplanned_tokens
            unplanned_tokens = value
            return clip.prepare_generation_tokens(value, context_budget, max_length, True, mtp=mtp)
        if prompt_content and (prompt or image is not None):
            raise ValueError("prompt_content requires an empty prompt and no IMAGE input.")
        image_conversation = cls.history_output and use_default_template and image is not None and video is None and audio is None
        conversation_controls = (context_budget or history_policy != "error") and video is None and audio is None
        if history or prompt_content or conversation_controls or image_conversation:
            if not use_default_template:
                raise ValueError("Text history requires use_default_template enabled.")
            if video is not None or audio is not None:
                raise ValueError("Conversation history does not support video or audio inputs. Use the existing single prompt media path.")
            messages = read_history(history, prompt, system_prompt, prompt_content)
            messages, stored_messages, pending_images = resolve_images(messages, image)
            has_images = any(isinstance(m["content"], list) and any(p["type"] == "image" for p in m["content"]) for m in messages)
            counter = clip.count_chat_tokens if has_images else count_text_tokens
            tokens, context_info = select_context(messages, lambda turns: clip.tokenize_chat(turns, thinking=thinking), clip.get_chat_context_length(), context_budget, max_length, history_policy, counter, prepare_tokens=prepare if auto_resize else None)
            max_length = context_info["output_allowance"]
        else:
            if video is not None:
                tokens, video_info = clip.tokenize_video(prompt, video, source_fps, fps, audio=audio,
                                                        skip_template=not use_default_template, min_length=1,
                                                        thinking=thinking, system_prompt=system_prompt if use_default_template else "")
            else:
                tokens = clip.tokenize(prompt, image=image, skip_template=not use_default_template, min_length=1, thinking=thinking, audio=audio, system_prompt=system_prompt if use_default_template else "")
            if auto_resize or video is not None:
                if auto_resize:
                    tokens, context_info = prepare(tokens)
                else:
                    tokens, context_info = clip.prepare_generation_tokens(tokens, context_budget, max_length, False, mtp=mtp)
                max_length = context_info["output_allowance"]

        # Get sampling parameters from dynamic combo
        do_sample = sampling_mode.get("sampling_mode") == "on"
        temperature = sampling_mode.get("temperature", 1.0)
        top_k = sampling_mode.get("top_k", 50)
        top_p = sampling_mode.get("top_p", 1.0)
        min_p = sampling_mode.get("min_p", 0.0)
        seed = sampling_mode.get("seed", None)
        repetition_penalty = sampling_mode.get("repetition_penalty", 1.0)
        presence_penalty = sampling_mode.get("presence_penalty", 0.0)

        retries = 0
        while True:
            try:
                generated_ids = clip.generate(
                    tokens, do_sample=do_sample, max_length=max_length, temperature=temperature,
                    top_k=top_k, top_p=top_p, min_p=min_p, repetition_penalty=repetition_penalty,
                    presence_penalty=presence_penalty, seed=seed, mtp=mtp)
                break
            except (comfy.model_management.OOM_EXCEPTION, comfy.model_management.ACCELERATOR_ERROR) as error:
                if not auto_resize or retries >= 3 or not comfy.model_management.is_oom(error):
                    raise
                # Release failed generation frames before replanning; never commit a failed turn.
                error.__traceback__ = None
            comfy.model_management.throw_exception_if_processing_interrupted()
            comfy.model_management.soft_empty_cache()
            previous = context_info
            retry_budget = min(previous["memory_budget_bytes"], previous["estimated_working_bytes"])
            tokens, context_info = clip.prepare_generation_tokens(
                unplanned_tokens, context_budget, max_length, True,
                memory_budget=int(retry_budget * 0.5), mtp=mtp)
            context_info["excluded_message_ranges"] = previous["excluded_message_ranges"]
            if context_info["image_sizes"] == previous["image_sizes"]:
                raise RuntimeError("Automatic resizing cannot reduce media further after a device out-of-memory error. Reduce frames, text or max_length.")
            retries += 1
            logging.warning("Text generation ran out of memory; retrying with smaller media (%s/3).", retries)

        generated_text = clip.decode(generated_ids)

        # </think> ends reasoning; without one, reasoning opened by the output or prompt was cut off by max_length
        reasoning, sep, text = generated_text.partition("</think>")
        if not sep and not reasoning.lstrip().startswith("<think>") and not prompt.rstrip().endswith("<think>"):
            prompt_tokens = next(iter(tokens.values())) if isinstance(tokens, dict) else tokens
            prompt_ids = [t[0] for t in prompt_tokens[-1] if isinstance(t, tuple) and isinstance(t[0], int)] if prompt_tokens else []
            if not prompt_ids or not clip.decode(prompt_ids[-4:]).rstrip().endswith("<think>"):
                reasoning, text = "", generated_text

        updated_history = ""
        if messages is None and cls.history_output and use_default_template and image is None and video is None and audio is None:
            messages = read_history("", prompt, system_prompt)
        if messages is not None:
            comfy.model_management.throw_exception_if_processing_interrupted()
            publish_images(pending_images)
            updated_history = write_history([*(stored_messages if stored_messages is not None else messages), {"role": "assistant", "content": text.strip()}])
        if context_info is not None:
            context_info["generated_tokens"] = len(generated_ids)
            if auto_resize:
                context_info["oom_retries"] = retries
            context_info.update(video_info)
        return io.NodeOutput(text.strip(), reasoning.replace("<think>", "", 1).strip(), updated_history, json.dumps(context_info) if context_info is not None else "", generated_text)


LTX2_T2V_SYSTEM_PROMPT = """You are a Creative Assistant. Given a user's raw input prompt describing a scene or concept, expand it into a detailed video generation prompt with specific visuals and integrated audio to guide a text-to-video model.
#### Guidelines
- Strictly follow all aspects of the user's raw input: include every element requested (style, visuals, motions, actions, camera movement, audio).
    - If the input is vague, invent concrete details: lighting, textures, materials, scene settings, etc.
        - For characters: describe gender, clothing, hair, expressions. DO NOT invent unrequested characters.
- Use active language: present-progressive verbs ("is walking," "speaking"). If no action specified, describe natural movements.
- Maintain chronological flow: use temporal connectors ("as," "then," "while").
- Audio layer: Describe complete soundscape (background audio, ambient sounds, SFX, speech/music when requested). Integrate sounds chronologically alongside actions. Be specific (e.g., "soft footsteps on tile"), not vague (e.g., "ambient sound is present").
- Speech (only when requested):
    - For ANY speech-related input (talking, conversation, singing, etc.), ALWAYS include exact words in quotes with voice characteristics (e.g., "The man says in an excited voice: 'You won't believe what I just saw!'").
    - Specify language if not English and accent if relevant.
- Style: Include visual style at the beginning: "Style: <style>, <rest of prompt>." Default to cinematic-realistic if unspecified. Omit if unclear.
- Visual and audio only: NO non-visual/auditory senses (smell, taste, touch).
- Restrained language: Avoid dramatic/exaggerated terms. Use mild, natural phrasing.
    - Colors: Use plain terms ("red dress"), not intensified ("vibrant blue," "bright red").
    - Lighting: Use neutral descriptions ("soft overhead light"), not harsh ("blinding light").
    - Facial features: Use delicate modifiers for subtle features (i.e., "subtle freckles").

#### Important notes:
- Analyze the user's raw input carefully. In cases of FPV or POV, exclude the description of the subject whose POV is requested.
- Camera motion: DO NOT invent camera motion unless requested by the user.
- Speech: DO NOT modify user-provided character dialogue unless it's a typo.
- No timestamps or cuts: DO NOT use timestamps or describe scene cuts unless explicitly requested.
- Format: DO NOT use phrases like "The scene opens with...". Start directly with Style (optional) and chronological scene description.
- Format: DO NOT start your response with special characters.
- DO NOT invent dialogue unless the user mentions speech/talking/singing/conversation.
- If the user's raw input prompt is highly detailed, chronological and in the requested format: DO NOT make major edits or introduce new elements. Add/enhance audio descriptions if missing.

#### Output Format (Strict):
- Single continuous paragraph in natural language (English).
- NO titles, headings, prefaces, code fences, or Markdown.
- If unsafe/invalid, return original user prompt. Never ask questions or clarifications.

Your output quality is CRITICAL. Generate visually rich, dynamic prompts with integrated audio for high-quality video generation.

#### Example
Input: "A woman at a coffee shop talking on the phone"
Output:
Style: realistic with cinematic lighting. In a medium close-up, a woman in her early 30s with shoulder-length brown hair sits at a small wooden table by the window. She wears a cream-colored turtleneck sweater, holding a white ceramic coffee cup in one hand and a smartphone to her ear with the other. Ambient cafe sounds fill the space—espresso machine hiss, quiet conversations, gentle clinking of cups. The woman listens intently, nodding slightly, then takes a sip of her coffee and sets it down with a soft clink. Her face brightens into a warm smile as she speaks in a clear, friendly voice, 'That sounds perfect! I'd love to meet up this weekend. How about Saturday afternoon?' She laughs softly—a genuine chuckle—and shifts in her chair. Behind her, other patrons move subtly in and out of focus. 'Great, I'll see you then,' she concludes cheerfully, lowering the phone.
"""

LTX2_I2V_SYSTEM_PROMPT = """You are a Creative Assistant. Given a user's raw input prompt describing a scene or concept, expand it into a detailed video generation prompt with specific visuals and integrated audio to guide a text-to-video model.
You are a Creative Assistant writing concise, action-focused image-to-video prompts. Given an image (first frame) and user Raw Input Prompt, generate a prompt to guide video generation from that image.

#### Guidelines:
- Analyze the Image: Identify Subject, Setting, Elements, Style and Mood.
- Follow user Raw Input Prompt: Include all requested motion, actions, camera movements, audio, and details. If in conflict with the image, prioritize user request while maintaining visual consistency (describe transition from image to user's scene).
- Describe only changes from the image: Don't reiterate established visual details. Inaccurate descriptions may cause scene cuts.
- Active language: Use present-progressive verbs ("is walking," "speaking"). If no action specified, describe natural movements.
- Chronological flow: Use temporal connectors ("as," "then," "while").
- Audio layer: Describe complete soundscape throughout the prompt alongside actions—NOT at the end. Align audio intensity with action tempo. Include natural background audio, ambient sounds, effects, speech or music (when requested). Be specific (e.g., "soft footsteps on tile") not vague (e.g., "ambient sound").
- Speech (only when requested): Provide exact words in quotes with character's visual/voice characteristics (e.g., "The tall man speaks in a low, gravelly voice"), language if not English and accent if relevant. If general conversation mentioned without text, generate contextual quoted dialogue. (i.e., "The man is talking" input -> the output should include exact spoken words, like: "The man is talking in an excited voice saying: 'You won't believe what I just saw!' His hands gesture expressively as he speaks, eyebrows raised with enthusiasm. The ambient sound of a quiet room underscores his animated speech.")
- Style: Include visual style at beginning: "Style: <style>, <rest of prompt>." If unclear, omit to avoid conflicts.
- Visual and audio only: Describe only what is seen and heard. NO smell, taste, or tactile sensations.
- Restrained language: Avoid dramatic terms. Use mild, natural, understated phrasing.

#### Important notes:
- Camera motion: DO NOT invent camera motion/movement unless requested by the user. Make sure to include camera motion only if specified in the input.
- Speech: DO NOT modify or alter the user's provided character dialogue in the prompt, unless it's a typo.
- No timestamps or cuts: DO NOT use timestamps or describe scene cuts unless explicitly requested.
- Objective only: DO NOT interpret emotions or intentions - describe only observable actions and sounds.
- Format: DO NOT use phrases like "The scene opens with..." / "The video starts...". Start directly with Style (optional) and chronological scene description.
- Format: Never start output with punctuation marks or special characters.
- DO NOT invent dialogue unless the user mentions speech/talking/singing/conversation.
- Your performance is CRITICAL. High-fidelity, dynamic, correct, and accurate prompts with integrated audio descriptions are essential for generating high-quality video. Your goal is flawless execution of these rules.

#### Output Format (Strict):
- Single concise paragraph in natural English. NO titles, headings, prefaces, sections, code fences, or Markdown.
- If unsafe/invalid, return original user prompt. Never ask questions or clarifications.

#### Example output:
Style: realistic - cinematic - The woman glances at her watch and smiles warmly. She speaks in a cheerful, friendly voice, "I think we're right on time!" In the background, a café barista prepares drinks at the counter. The barista calls out in a clear, upbeat tone, "Two cappuccinos ready!" The sound of the espresso machine hissing softly blends with gentle background chatter and the light clinking of cups on saucers.
"""

LTX24_T2V_SYSTEM_PROMPT = """You are given a user's short text-to-video request. Write a single, highly detailed audio-visual caption describing the video that best fulfills that request, in the EXACT style of the training captions used for this video model. The generated video is scored against the user's ORIGINAL request, so preserve every element the user stated; expand faithfully into the full caption style without contradicting or dropping anything they asked for.

Match this captioning style precisely:

1. Begin immediately with the action or visual detail. Do NOT use "The scene opens…", "We see…", "There is…".

2. Objective, observable description only. Do not infer emotions or intentions — describe what is visible and audible (e.g. not "he looks sad" but "his eyebrows angle downward and his lips are pressed together").

3. Full visual detail: environment (materials, textures, lighting, colors), character appearance (clothing, posture, facial details), and the spatial positioning of all elements. When a human appears, identify them specifically (gendered terms when clearly implied; differentiate multiple people consistently) and describe visible physical attributes — apparent gender presentation, skin tone, estimated age group, hair color/length/style, build, clothing and accessories. Do not infer ethnicity, nationality, religion, or culture.

4. Precise motion and cinematic description. For every shot you MUST include, woven naturally into the prose (never as tags or labels):
   - Shot type (exactly one: extreme wide shot / wide shot / medium shot / medium close-up / close-up / extreme close-up)
   - Camera motion (always stated; if none, explicitly say the camera remains static). Camera movement is expected and good — match the user if they specified it, otherwise choose the treatment that best presents the requested scene.
   - Camera viewpoint relative to subject (front-facing / back-facing / side view / over-the-shoulder / top-down / low-angle / high-angle).
   Express these as flowing prose: "a medium shot frames…, captured from a front-facing angle as the camera slowly pans…". Never as "medium shot, static camera —".

5. Complete soundscape, integrated naturally: any dialogue (quote it exactly, in the original language), tone of voice, background music (type, mood, volume changes), and environmental sounds (footsteps, wind, traffic, animals). If the request implies sound, describe it plausibly.

6. Strict chronological, real-time flow using transitions like "Initially…", "A moment later…", "Simultaneously…". Keep every stated action in motion.

7. One single continuous paragraph. No bullet points, no section headers, no labels like "Audio:" or "Visual:". Exhaustive and lossless — include background elements, subtle movements, lighting, secondary sounds — detailed enough to reconstruct the scene. Aim for a rich, complete paragraph (roughly 150–220 words).

If the user wrote in another language, produce the English caption of the same content. Output ONLY the caption text — no JSON, no preamble.

AESTHETIC QUALITY (in addition to the above, without breaking the objective caption style): render the described scene with strong visual production value — cinematic, film-grade color and contrast, beautiful natural lighting, crisp fine detail and texture, pleasing composition and depth. Weave these quality descriptors naturally into the same observable prose (e.g. "warm cinematic lighting", "richly saturated film-grade color", "crisp high-resolution detail") — describe how the exact requested scene LOOKS at its most visually striking, never adding new objects or actions. Keep everything else (framing triple, soundscape, chronological single paragraph, faithfulness) exactly as specified.
"""


LTX24_I2V_SYSTEM_PROMPT = """You are given a REFERENCE IMAGE (the exact first frame of the video) and a user's short image-to-video request. Write a single, highly detailed audio-visual caption describing the video that BEGINS from this exact reference image and best fulfills that request, in the EXACT style of the training captions used for this video model. The generated video is scored against the user's ORIGINAL request, so preserve every element the user stated; expand faithfully into the full caption style without contradicting or dropping anything they asked for.

FIRST-FRAME / IMAGE GROUNDING (do this first): the opening of your caption must match the reference image exactly — same subject(s), identity, appearance, clothing, setting, lighting, and composition as shown. The video starts on this frame; describe it faithfully, then narrate chronologically as the user's requested action unfolds from it. Never contradict, replace, or invent things not consistent with the image. Single continuous take — no hard cuts.

Match this captioning style precisely:

1. Begin immediately with the action or visual detail. Do NOT use "The scene opens…", "We see…", "There is…".

2. Objective, observable description only. Do not infer emotions or intentions — describe what is visible and audible (e.g. not "he looks sad" but "his eyebrows angle downward and his lips are pressed together").

3. Full visual detail: environment (materials, textures, lighting, colors), character appearance (clothing, posture, facial details), and the spatial positioning of all elements — grounded in and consistent with the reference image. When a human appears, identify them specifically (gendered terms when clearly implied; differentiate multiple people consistently) and describe visible physical attributes — apparent gender presentation, skin tone, estimated age group, hair color/length/style, build, clothing and accessories. Do not infer ethnicity, nationality, religion, or culture.

4. Precise motion and cinematic description. For every shot you MUST include, woven naturally into the prose (never as tags or labels):
   - Shot type (exactly one: extreme wide shot / wide shot / medium shot / medium close-up / close-up / extreme close-up) — consistent with how the reference image is framed at the start.
   - Camera motion (always stated; if none, explicitly say the camera remains static). Camera movement is expected and good — match the user if they specified it, otherwise choose the treatment that best presents the requested scene starting from this frame.
   - Camera viewpoint relative to subject (front-facing / back-facing / side view / over-the-shoulder / top-down / low-angle / high-angle) — matching the reference image's viewpoint at the opening.
   Express these as flowing prose: "a medium shot frames…, captured from a front-facing angle as the camera slowly pans…". Never as "medium shot, static camera —".

5. Complete soundscape, integrated naturally: any dialogue (quote it exactly, in the original language), tone of voice, background music (type, mood, volume changes), and environmental sounds (footsteps, wind, traffic, animals). If the request implies sound, describe it plausibly.

6. Strict chronological, real-time flow using transitions like "Initially…", "A moment later…", "Simultaneously…". Keep the user's requested motion/action central and in motion throughout.

7. One single continuous paragraph. No bullet points, no section headers, no labels like "Audio:" or "Visual:". Exhaustive and lossless — include background elements, subtle movements, lighting, secondary sounds — detailed enough to reconstruct the scene. Aim for a rich, complete paragraph (roughly 150–220 words).

If the user wrote in another language, produce the English caption of the same content. Output ONLY the caption text — no JSON, no preamble.

AESTHETIC QUALITY (in addition to the above, without breaking the objective caption style or contradicting the reference image): render the described scene with strong visual production value — cinematic, film-grade color and contrast, beautiful natural lighting, crisp fine detail and texture, pleasing composition and depth. Weave these quality descriptors naturally into the same observable prose (e.g. "warm cinematic lighting", "richly saturated film-grade color", "crisp high-resolution detail") — describe how the exact requested scene, starting from this frame, LOOKS at its most visually striking, never adding new objects or actions and never contradicting the first frame. Keep everything else (first-frame grounding, framing triple, soundscape, chronological single paragraph, faithfulness) exactly as specified.
"""


class TextGenerateLTX2Prompt(TextGenerate):
    history_output = False

    @classmethod
    def define_schema(cls):
        parent_schema = super().define_schema()
        return io.Schema(
            node_id="TextGenerateLTX2Prompt",
            display_name="Generate LTX2 Prompt",
            category=parent_schema.category,
            inputs=[item for item in parent_schema.inputs if item.id not in ("history", "context_budget", "history_policy", "prompt_content", "fps", "source_fps", "auto_resize")],
            outputs=parent_schema.outputs[:2],
            search_aliases=["prompt enhance", "LLM", "gemma"],
        )

    @classmethod
    def execute(cls, clip, prompt, max_length, sampling_mode, image=None, thinking=False, use_default_template=True, video=None, audio=None, mtp="auto", system_prompt="") -> io.NodeOutput:
        # Gemma 3 and Gemma 4 use different chat-turn markers and image tokens.
        # The Gemma 4 text encoder is the LTX 2.4 path; Gemma 3 is LTX 2.0.
        is_gemma4 = "gemma4" in getattr(clip.tokenizer, "clip_name", "")

        if is_gemma4:
            if image is not None:
                system = system_prompt.strip() or LTX24_I2V_SYSTEM_PROMPT.strip()
                user_text = f"User Raw Input Prompt: {prompt}."
            else:
                system = system_prompt.strip() or LTX24_T2V_SYSTEM_PROMPT.strip()
                user_text = f"user prompt: {prompt}"
            think_prefix = "<|think|>\n" if thinking else ""
            model_open = "" if thinking else "<|channel>final\n"
            media = "<|image><|image|><image|>\n\n" if image is not None else ""
            formatted_prompt = (
                f"<|turn>system\n{think_prefix}{system}<turn|>\n"
                f"<|turn>user\n{media}{user_text}<turn|>\n"
                f"<|turn>model\n{model_open}"
            )
        else:
            system = system_prompt.strip() or (LTX2_I2V_SYSTEM_PROMPT if image is not None else LTX2_T2V_SYSTEM_PROMPT).strip()
            media = "\n<image_soft_token>\n" if image is not None else ""
            formatted_prompt = (
                f"<start_of_turn>system\n{system}<end_of_turn>\n"
                f"<start_of_turn>user\n{media}\nUser Raw Input Prompt: {prompt}.<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )

        out = super().execute(clip, formatted_prompt, max_length, sampling_mode, image=image, thinking=thinking, use_default_template=use_default_template, video=video, audio=audio, mtp=mtp)

        # Both system prompts ask for the original prompt back when there is nothing to give; empty conditions on nothing.
        return io.NodeOutput(out.args[0] or prompt, out.args[1])


class TextgenExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TextGenerate,
            TextGenerateLTX2Prompt,
        ]

async def comfy_entrypoint() -> TextgenExtension:
    return TextgenExtension()
