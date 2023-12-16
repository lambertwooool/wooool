from .patch_manager import patch, undo

import modules.model.sd1_clip
import modules.prompt_helper

class ClipPatches:
    def __init__(self):
        self.Clip_token_weights = patch(__name__, modules.model.sd1_clip, 'token_weights', modules.prompt_helper.parse_prompt_attention)

    def undo(self):
        self.Clip_token_weights = undo(__name__, modules.model.sd1_clip, 'token_weights')