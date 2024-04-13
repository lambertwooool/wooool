import torch
from torch import nn

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from .configuration_uform_gen import VLMConfig
from .configuration_uform_gen import VLMConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from typing import List, Optional, Tuple, Union

class ImageFeaturesPooler(nn.Module):
    def __init__(self, config, text_config):
        super().__init__()
        self.pooler = nn.TransformerDecoderLayer(
            config.image_encoder_hidden_size,
            config.image_pooler_num_attn_heads,
            config.image_pooler_intermediate_size,
            activation=nn.functional.silu,
            batch_first=True,
            norm_first=True,
        )
        self.image_latents = nn.Parameter(
            torch.randn(1, config.num_image_latents, config.image_encoder_hidden_size)
            * config.initializer_range**0.5
        )
        self.projection = nn.Linear(config.image_encoder_hidden_size, text_config.hidden_size)

    def forward(self, features):
        features = self.pooler(
            self.image_latents.expand(features.size(0), -1, -1), features
        )

        return self.projection(features)


class VLMPreTrainedModel(PreTrainedModel):
    config_class = VLMConfig
    base_model_prefix = "vlm"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        pass

    def _initialize_weights(self, module):
        pass


class VLMForCausalLM(VLMPreTrainedModel):
    def __init__(self,
                 text_decoder: Qwen2ForCausalLM,
                 config: VLMConfig,
                 text_config: Qwen2Config):
        super().__init__(config)

        # self.text_decoder_wrap = text_decoder_wrap
        self.text_decoder = text_decoder
        self.config = config
        self.text_config = text_config

    def get_input_embeddings(self):
        return self.text_decoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_decoder.set_input_embeddings(value)

    def gather_continuous_embeddings(
        self,
        input_ids: torch.Tensor,
        word_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor
    ) -> torch.Tensor:
        
        start_indices = (input_ids == self.config.image_token_id).nonzero()[:, 1]
        embeddings = []
        for sample_idx, start_idx in enumerate(start_indices.tolist()):
            embeddings.append(
                torch.cat(
                    (
                        word_embeddings[sample_idx, :start_idx],
                        image_embeddings[sample_idx],
                        word_embeddings[sample_idx, start_idx + 1 :],
                    ),
                    dim=0,
                )
            )

        return torch.stack(embeddings, dim=0)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[dict, Tuple, CausalLMOutputWithPast]:        
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_is or inputs_embeds")

        if inputs_embeds is None and past_key_values is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            if images is not None:
                # image_embeds = self.get_images_embeddings(images)
                image_embeds = images
                inputs_embeds = self.gather_continuous_embeddings(
                    input_ids,
                    inputs_embeds,
                    image_embeds
                )

        if position_ids is None:
            seq_length = (
                inputs_embeds.shape[1]
                if inputs_embeds is not None
                else input_ids.shape[1]
            )
            past_key_values_length = 0

            if past_key_values is not None:
                past_key_values_length = past_key_values[0][0].shape[2]

            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        outputs = self.text_decoder(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids if past_key_values is not None else None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        images=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
            n_samples = inputs_embeds.shape[0]
        else:
            model_inputs = {"input_ids": input_ids}
            n_samples = input_ids.shape[0]

        if images is not None:
            model_inputs["images"] = images

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images if past_key_values is None else None,
            }
        )
        return model_inputs
