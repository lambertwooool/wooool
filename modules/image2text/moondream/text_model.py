import torch
import re
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import CodeGenTokenizerFast as Tokenizer
from .modeling_phi import PhiForCausalLM

class TextModel(torch.nn.Module):
    def __init__(self, tokenizer: Tokenizer, text_model: PhiForCausalLM) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.text_model = text_model

        # with init_empty_weights():
        #     self.model = PhiForCausalLM(phi_config)

        # self.model = load_checkpoint_and_dispatch(
        #     self.model,
        #     f"{model_path}/text_model.pt",
        #     device_map={"": device},
        #     dtype=dtype,
        # )
    
    def input_embeds(self, prompt: str, image_embeds: torch.Tensor, tokenizer: Tokenizer):
        def _tokenize(txt):
            return tokenizer(
                txt, return_tensors="pt", add_special_tokens=False
            ).input_ids

        text_emb = self.text_model.get_input_embeddings()

        # Add BOS token
        embeds = []
        embeds.append(
            text_emb((torch.tensor([[tokenizer.bos_token_id]], device=image_embeds.device)))
        )

        if "<image>" not in prompt:
            embeds.append(text_emb(_tokenize(prompt)))
        else:
            assert prompt.count("<image>") == 1
            before, after = prompt.split("<image>")
            if len(before) > 0:
                embeds.append(text_emb(_tokenize(before).to(image_embeds.device)))
            embeds.append(image_embeds)
            if len(after) > 0:
                embeds.append(text_emb(_tokenize(after).to(image_embeds.device)))

        return torch.cat(embeds, dim=1)

    # def generate(
    #     self,
    #     image_embeds: torch.Tensor,
    #     prompt: str,
    #     tokenizer: Tokenizer,
    #     max_new_tokens: int=128,
    #     **kwargs,
    # ):  
    #     generate_config = {
    #         "eos_token_id": tokenizer.eos_token_id,
    #         "bos_token_id": tokenizer.bos_token_id,
    #         "pad_token_id": tokenizer.bos_token_id,
    #         "max_new_tokens": max_new_tokens,
    #         **kwargs,
    #     }
        
    #     if self.eos_text:
    #         eos_token_id = self.tokenizer(self.eos_text, add_special_tokens=False)[0].ids
    #         generate_config["eos_token_id"] = eos_token_id

    #     with torch.inference_mode():
    #         inputs_embeds = self.input_embeds(prompt, image_embeds, tokenizer)
    #         output_ids = self.text_model.generate(
    #             inputs_embeds=inputs_embeds, **generate_config
    #         )

    #     return tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # def answer_question(
    #     self,
    #     image_embeds: torch.Tensor,
    #     question: str,
    #     tokenizer: Tokenizer=None,
    #     max_new_tokens: int=512,
    #     chat_history="",
    #     result_queue=None,
    #     **kwargs,
    # ):
    #     prompt = f"<image>\n\n{chat_history}Question: {question}\n\nAnswer:"
    #     tokenizer = self.tokenizer if tokenizer is None else tokenizer
    #     answer = self.generate(
    #         image_embeds,
    #         prompt,
    #         tokenizer=tokenizer,
    #         max_new_tokens=max_new_tokens,
    #         **kwargs,
    #     )[0]
    #     cleaned_answer = (re.sub(f"{self.eos_text[:1]}$|{self.eos_text[:-1]}$", "", answer) if self.eos_text else answer).strip()

    #     # Use the result_queue to pass the result if it is provided
    #     if result_queue:
    #         result_queue.put(cleaned_answer)
    #     else:
    #         return cleaned_answer

    # def batch_answer(
    #     self,
    #     images,
    #     prompts: str,
    #     tokenizer: Tokenizer=None,
    #     **kwargs,
    # ):
    #     image_embeds = self.encode_image(images)

    #     templated_prompts = [
    #         f"<image>\n\nQuestion: {prompt}\n\nAnswer:" for prompt in prompts
    #     ]
    #     prompt_embs = [
    #         self.input_embeds(prompt, image_embed.unsqueeze(0), tokenizer)[0]
    #         for prompt, image_embed in zip(templated_prompts, image_embeds)
    #     ]

    #     bos_emb = prompt_embs[0][0]
    #     max_len = max([p.shape[0] for p in prompt_embs])

    #     inputs_embeds = torch.cat(
    #         [
    #             torch.cat([bos_emb.repeat(max_len - p.shape[0], 1), p]).unsqueeze(0)
    #             for p in prompt_embs
    #         ],
    #         dim=0,
    #     )
    #     attention_mask = torch.cat(
    #         [
    #             torch.cat(
    #                 [
    #                     torch.zeros(
    #                         1,
    #                         max_len - p.shape[0],
    #                         device=image_embeds.device,
    #                         dtype=torch.long,
    #                     ),
    #                     torch.ones(1, p.shape[0], device=image_embeds.device, dtype=torch.long),
    #                 ],
    #                 dim=1,
    #             )
    #             for p in prompt_embs
    #         ],
    #         dim=0,
    #     )

    #     tokenizer = self.tokenizer if tokenizer is None else tokenizer
    #     generate_config = {
    #         "eos_token_id": tokenizer.eos_token_id,
    #         "bos_token_id": tokenizer.bos_token_id,
    #         "pad_token_id": tokenizer.bos_token_id,
    #         "max_new_tokens": 512,
    #         **kwargs,
    #     }

    #     with torch.no_grad():
    #         output_ids = self.text_model.generate(
    #             inputs_embeds=inputs_embeds,
    #             attention_mask=attention_mask,
    #             **generate_config,
    #         )

    #     return [
    #         x.strip()
    #         for x in tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    #     ]