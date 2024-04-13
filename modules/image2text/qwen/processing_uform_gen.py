from functools import partial

import torch
import torch.nn.functional as F
from transformers.processing_utils import ProcessorMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers import AutoTokenizer, AutoConfig
from transformers import BatchFeature
from transformers import CodeGenTokenizerFast as Tokenizer
from .configuration_uform_gen import VLMConfig

from PIL import Image
from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
    ToTensor
)


IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)


def convert_to_rgb(x):
    return x.convert("RGB")


def expand2square(image, background_color):
    width, height = image.size
    if width == height:
        return image
    elif width > height:
        result = Image.new(image.mode, (width, width), background_color)
        result.paste(image, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(image.mode, (height, height), background_color)
        result.paste(image, ((height - width) // 2, 0))
        return result


class ImageProcessor(BaseImageProcessor):
    def __init__(
        self,
        image_size: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.transform = Compose(
            [
                convert_to_rgb,
                partial(
                    expand2square,
                    background_color=tuple(int(255 * v) for v in IMAGENET_MEAN)
                ),
                Resize(image_size),
                ToTensor(),
                Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                ),
            ]
        )
    
    def preprocess(
        self,
        image: Image
    ):
        return self.transform(image)

    def __repr__(self):
        return repr(self.transform)


class VLMProcessor(ProcessorMixin):
    def __init__(self, tokenizer: Tokenizer, config: VLMConfig):
        self.config = config
        self.image_size = config.image_size
        
        self.feature_extractor = ImageProcessor(self.image_size)
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     config.text_decoder_name_or_path, additional_special_tokens=["<image>"]
        # )
        self.tokenizer = tokenizer
        self.tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        self.num_image_latents = config.num_image_latents
        # super().__init__(self.image_processor, self.tokenizer)

    def __call__(
        self, text=None, images=None, return_tensors="pt", **kwargs
    ):
        if text is not None:
            if isinstance(text, str):
                text = [text]

            tokenized_texts = []
            for t in text:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f" <image> {t}"},
                ]
                tokenized_prompt = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors=return_tensors
                )

                tokenized_texts.append(tokenized_prompt)

            max_len = max(len(t[0]) for t in tokenized_texts)
            input_ids = torch.full(
                (len(tokenized_texts), max_len),
                fill_value=self.tokenizer.pad_token_id,
                dtype=torch.int64,
            )
            attention_mask = torch.full(
                (len(tokenized_texts), max_len), fill_value=0, dtype=torch.int64
            )

            for i, tokens in enumerate(tokenized_texts):
                input_ids[i, -len(tokens[0]) :] = tokens[0]
                attention_mask[i, -len(tokens[0]) :] = 1

            attention_mask = F.pad(
                attention_mask, pad=(0, self.num_image_latents - 1), value=1
            )

            encoding = BatchFeature(
                data={"input_ids": input_ids, "attention_mask": attention_mask}
            )

        if images is not None:
            if isinstance(images, (list, tuple)):
                image_features = torch.empty(
                    (len(images), 3, self.image_size , self.image_size),
                    dtype=torch.float32,
                )

                for i, image in enumerate(images):
                    image_features[i] = self.feature_extractor(image)

            else:
                image_features = self.image_processor(images).unsqueeze(0)

        if text is not None and images is not None:
            encoding["images"] = image_features
            return encoding

        elif text is not None:
            return encoding

        else:
            return BatchFeature(
                data={
                    "images": image_features,
                },
                tensor_type=return_tensors,
            )

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    # @classmethod
    # def from_pretrained(
    #     cls,
    #     pretrained_model_name_or_path,
    #     trust_remote_code=False,
    #     **kwargs
    # ):
    #     config = AutoConfig.from_pretrained(
    #         pretrained_model_name_or_path,
    #         trust_remote_code=trust_remote_code
    #     )
    #     return cls(config)
