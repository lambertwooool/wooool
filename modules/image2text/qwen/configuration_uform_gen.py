from transformers.configuration_utils import PretrainedConfig
from typing import List


class VLMConfig(PretrainedConfig):
    model_type = "vlm"

    def __init__(
        self,
        text_decoder_name_or_path: str = "",
        image_encoder_name_or_path: str = "",
        image_size: int = 336,
        image_pooler_num_attn_heads: int = 16,
        image_pooler_intermediate_size: int = 3200,
        image_token_id: int = 151646,
        image_encoder_hidden_size: int = 1280,
        image_encoder_patch_size: int = 14,
        image_encoder_num_layers: int = 32,
        image_encoder_num_heads: int = 16,
        image_encoder_pooling: str = "cls",
        num_image_latents: int = 256,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        **kwargs,
    ):
        self.text_decoder_name_or_path = text_decoder_name_or_path
        self.image_encoder_name_or_path = image_encoder_name_or_path

        self.image_pooler_num_attn_heads = image_pooler_num_attn_heads
        self.image_pooler_intermediate_size = image_pooler_intermediate_size
        self.image_token_id = image_token_id
        self.image_size = image_size
        self.image_encoder_hidden_size = image_encoder_hidden_size
        self.image_encoder_patch_size = image_encoder_patch_size
        self.image_encoder_num_layers = image_encoder_num_layers
        self.image_encoder_num_heads = image_encoder_num_heads
        self.image_encoder_pooling = image_encoder_pooling
        self.num_image_latents = num_image_latents

        self.initializer_range = initializer_range
        self.use_cache = use_cache
        
        super().__init__(**kwargs)