import os
import re
import random
from dynamicprompts.generators import RandomPromptGenerator
from dynamicprompts.wildcards.wildcard_manager import WildcardManager

import modules.paths
from modules import lora, localization
from modules.util import load_json, dictJoin, listJoin, list_files

wm = WildcardManager(modules.paths.wildcard_path)
dynamic_generator = RandomPromptGenerator(wildcard_manager=wm)
re_attention = re.compile(r"[\(\)\[\]]|:\s*([+-]?[.\d]+)\s*[\)\]]|[^\(\)\[\]:]+", re.X)
re_normalize = [re.compile(x, re.X) for x in [r"([ ])\s+", r"\s*([,\.])\s*", r"([,\.])[,\.]+"]]
re_random_style = re.compile(r"^__style_(\w+)__$")

def generate(main_character_prompt, prompt_main, prompt_negative, batch_size, style, style_weight, options, loras, lang, seeds, sd_type="sdxl", skip_prompt=False):
    if not skip_prompt:
        mc_name, mc_prompt = main_character_prompt
        lang_desc = localization.localization_json(lang).get("@lang-desc", "") if lang else ""
        options = { "main_character": mc_prompt, **options }
        for k, v in options.items():
            if isinstance(v, dict):
                prompt_negative += ", " + v.get("negative_prompt", "")
                pm_weight = v.get("weight", 1.0)
                loras += get_config_loras(v, sd_type, pm_weight)
                v = v.get("prompt", "")
                if pm_weight != 1.0 and v != "":
                    v = f"({v}:{pm_weight})"
            options[k] = v.replace("{lang}", f"{lang_desc}")

        positive = {
            "main_character": options.pop("main_character"),
            "prompt_main": prompt_main,
            **options,
            "prompt_lora": listJoin([x[2] for x in loras])
        }
        positive = dictJoin(positive)

        batch_size = min(8, batch_size)
        positive, negative, style, style_loras = apply_style(style, style_weight, positive, prompt_negative, sd_type=sd_type, seeds=seeds)
        loras += style_loras
    else:
        positive, negative = prompt_main, prompt_negative
    
    positive, lora_blocks = lora.parse_block(positive)
    kw_loras = lora.keyword_loras(positive)
    loras = lora.reduce(loras + lora_blocks + kw_loras)

    positive = [normalize(positive)] + dynamic_prompt(positive, num_images=batch_size, seeds=seeds)
    negative = [normalize(negative)] + dynamic_prompt(negative, num_images=batch_size, seeds=seeds)
    
    return positive, negative, style, loras

def re_generate(positive_template, negative_template, batch_size=8):
    batch_size = min(8, batch_size)
    positive = dynamic_prompt(positive_template, num_images=batch_size)
    negative = dynamic_prompt(negative_template, num_images=batch_size)

    return positive, negative

def get_config_loras(config, sd_type, weight=1.0):
    loras = []
    item_loras = config.get(f"loras_{sd_type}", {})
    # print(config, item_loras, weight)
    for lk, lv in item_loras.items():
        kw = ""
        if isinstance(lv, list):
            kw = lv[1]
            lv = lv[0]
        loras.append((lk, round(lv * weight, 2), kw))
    return loras

def dynamic_prompt(positive, num_images=1, seeds=None, lang=None):
    if lang is not None:
        lang_desc = localization.localization_json(lang).get("@lang-desc", "") if lang else ""
        positive = positive.replace("{lang}", f"{lang_desc}")

    if positive != "":
        return [[normalize(x)] for x in dynamic_generator.generate(positive, num_images=num_images, seeds=seeds)]
    else:
        return [[""] for _ in range(num_images)]

def apply_style(style, style_weight, positive, negative, sd_type="sdxl", seeds=None):
    if style is not None:
        template, style_name = find_style(style, seeds)
        if template is not None:
            prompt = template.get("prompt", "{prompt}").split("{prompt}")
            style_loras = get_config_loras(template, sd_type, style_weight)

            # positive = template.get("prompt", "{prompt}").replace('{prompt}', positive)
            if style_weight != 1.0:
                prompt = [f"({x.strip()}:{style_weight}) " for x in prompt if x != ""]
            positive = positive.join([x for x in prompt if x != ""]).strip()

            negative = listJoin([negative, template["negative_prompt"]]).strip()

    return positive, negative, style_name, style_loras

def find_style(style_name, seeds=None):
    style_type = re.search(re_random_style, style_name)
    if style_type is not None:
        style_type = style_type.groups()[0]
        file = os.path.join(modules.paths.sd_style_path, f"{style_type}.json")
        data = load_json(file)
        styles = "|".join([x["name"] for x in data])
        style_name = dynamic_prompt(f"{{{styles}}}", num_images=len(seeds), seeds=seeds)[0][0]
        # template = data[random.randint(0, len(styles))]
        template = list(filter(lambda x: x["name"] == style_name, data))
        return template[0], style_name
    else:
        files = list_files(modules.paths.sd_style_path, "json")
        for file in files:
            data = load_json(file)
            template = list(filter(lambda x: x["name"] == style_name, data))
            if template:
                return template[0], style_name
    return None, None

# patch to clip.parse_parentheses
def parse_prompt_attention(prompt, current_weight=1.0):
    res = []
    round_brackets = []
    round_bracket_mul = 1.1
    square_brackets = []
    square_bracket_mul = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(prompt):
        text = m.group(0)
        weight = m.group(1)

        # round bracket
        if text == "(":
            round_brackets.append(len(res))
        elif text == ')' and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_mul)
        elif weight is not None and text.endswith(')') and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))

        # square bracket
        elif text == '[':
            square_brackets.append(len(res))
        elif text == ']' and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_mul)
        elif weight is not None and text.endswith(']') and square_brackets:
            multiply_range(square_brackets.pop(), float(weight))

        else:
            res.append([text, current_weight])
        
    for pos in round_brackets:
        multiply_range(pos, round_bracket_mul)

    for pos in square_brackets:
        multiply_range(pos, square_brackets)

    return res


def normalize(prompt):
    for re_n in re_normalize:
        prompt = re.sub(re_n, "\\1", prompt)

    return prompt
