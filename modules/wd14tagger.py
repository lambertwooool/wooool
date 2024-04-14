# https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags
from modules.image2text import Image2TextProcessor

default_exclude_tags = []

def tag(image, threshold=0.35, character_threshold=0.85, exclude_tags=[], only_text=True):
    process = Image2TextProcessor("wd14_v3")
    
    output = process(   image,
                        threshold=threshold,
                        character_threshold=character_threshold,
                        exclude_tags=exclude_tags,
                        only_text=only_text,
                    )
    
    if only_text:
        output = output.split(",")

    return output