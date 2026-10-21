import html
import re

import torch
from tokenizers import Tokenizer

from comfy import sd1_clip
from comfy.text_encoders.llama import Qwen3_06B

try:
    import ftfy
except ImportError:
    ftfy = None


class MossSoundEffectTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        if ftfy is None:
            raise ImportError('MOSS SoundEffect prompt cleaning requires ftfy; install requirements-moss-soundeffect.txt in the configured runtime')
        self.tokenizer_json = tokenizer_data['tokenizer_json']
        if torch.is_tensor(self.tokenizer_json):
            self.tokenizer_json = self.tokenizer_json.numpy().tobytes()
        self.tokenizer = Tokenizer.from_str(self.tokenizer_json.decode())

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        text = re.sub(r'\s+', ' ', html.unescape(html.unescape(ftfy.fix_text(text)))).strip()
        ids = self.tokenizer.encode(text, add_special_tokens=True).ids
        ids += [-1] * max(0, 512 - len(ids))
        return {'moss_soundeffect': [[(token, 1.0, i) if return_word_ids else (token, 1.0) for i, token in enumerate(ids)]]}

    def state_dict(self):
        return {'tokenizer_json': torch.frombuffer(bytearray(self.tokenizer_json), dtype=torch.uint8)}

    def decode(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


class MossSoundEffectClipModel(sd1_clip.SDClipModel):
    def __init__(self, device='cpu', dtype=None, model_options={}):
        config = dict(hidden_size=2048, intermediate_size=6144, max_position_embeddings=40960)
        super().__init__(device=device, dtype=dtype, textmodel_json_config=config, model_class=Qwen3_06B,
                         special_tokens={'pad': -1}, layer_norm_hidden_state=False, enable_attention_masks=True,
                         zero_out_masked=True, model_options=model_options)
        self.can_assign_sd = False

    def process_tokens(self, tokens, device):
        lengths = [sum(token != -1 for token in row) for row in tokens]
        ids = torch.tensor(tokens, device=device, dtype=torch.long)
        mask = ids != -1
        embeds = self.transformer.get_input_embeddings()(ids.masked_fill(~mask, 151643), out_dtype=self.transformer.dtype)
        return embeds, mask, lengths, []

    def load_sd(self, state_dict):
        state_dict = {key: value for key, value in state_dict.items() if key != 'model.lm_head.weight'}
        return self.transformer.load_state_dict(state_dict, strict=True, assign=self.can_assign_sd)


def te(dtype_llama=None, llama_quantization_metadata=None):
    class MossSoundEffectTEModel(sd1_clip.SD1ClipModel):
        def __init__(self, device='cpu', dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = {**model_options, 'quantization_metadata': llama_quantization_metadata}
            super().__init__(device=device, dtype=dtype_llama or dtype, name='moss_soundeffect', clip_model=MossSoundEffectClipModel, model_options=model_options)
    return MossSoundEffectTEModel
