from modules.model import model_helper, supported_models, supported_models_base

def model_config_from_unet_config(unet_config):
    for model_config in supported_models.models:
        if model_config.matches(unet_config):
            return model_config(unet_config)

    print("no match", unet_config)
    return None

def model_config_from_unet(state_dict, unet_key_prefix, dtype, use_base_if_no_match=False):
    unet_config = model_helper.detect_unet_config(state_dict, unet_key_prefix, dtype)
    model_config = model_config_from_unet_config(unet_config)
    if model_config is None and use_base_if_no_match:
        return supported_models_base.BASE(unet_config)
    else:
        return model_config

def unet_config_from_diffusers_unet(state_dict, dtype):
    match = {}
    attention_resolutions = []

    attn_res = 1
    for i in range(5):
        k = "down_blocks.{}.attentions.1.transformer_blocks.0.attn2.to_k.weight".format(i)
        if k in state_dict:
            match["context_dim"] = state_dict[k].shape[1]
            attention_resolutions.append(attn_res)
        attn_res *= 2

    match["attention_resolutions"] = attention_resolutions

    match["model_channels"] = state_dict["conv_in.weight"].shape[0]
    match["in_channels"] = state_dict["conv_in.weight"].shape[1]
    match["adm_in_channels"] = None
    if "class_embedding.linear_1.weight" in state_dict:
        match["adm_in_channels"] = state_dict["class_embedding.linear_1.weight"].shape[1]
    elif "add_embedding.linear_1.weight" in state_dict:
        match["adm_in_channels"] = state_dict["add_embedding.linear_1.weight"].shape[1]

    SDXL = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
            'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
            'num_res_blocks': 2, 'attention_resolutions': [2, 4], 'transformer_depth': [0, 2, 10], 'channel_mult': [1, 2, 4],
            'transformer_depth_middle': 10, 'use_linear_in_transformer': True, 'context_dim': 2048, "num_head_channels": 64}

    SDXL_refiner = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                    'num_classes': 'sequential', 'adm_in_channels': 2560, 'dtype': dtype, 'in_channels': 4, 'model_channels': 384,
                    'num_res_blocks': 2, 'attention_resolutions': [2, 4], 'transformer_depth': [0, 4, 4, 0], 'channel_mult': [1, 2, 4, 4],
                    'transformer_depth_middle': 4, 'use_linear_in_transformer': True, 'context_dim': 1280, "num_head_channels": 64}

    SD21 = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
            'adm_in_channels': None, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320, 'num_res_blocks': 2,
            'attention_resolutions': [1, 2, 4], 'transformer_depth': [1, 1, 1, 0], 'channel_mult': [1, 2, 4, 4],
            'transformer_depth_middle': 1, 'use_linear_in_transformer': True, 'context_dim': 1024, "num_head_channels": 64}

    SD21_uncliph = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                    'num_classes': 'sequential', 'adm_in_channels': 2048, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
                    'num_res_blocks': 2, 'attention_resolutions': [1, 2, 4], 'transformer_depth': [1, 1, 1, 0], 'channel_mult': [1, 2, 4, 4],
                    'transformer_depth_middle': 1, 'use_linear_in_transformer': True, 'context_dim': 1024, "num_head_channels": 64}

    SD21_unclipl = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                    'num_classes': 'sequential', 'adm_in_channels': 1536, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
                    'num_res_blocks': 2, 'attention_resolutions': [1, 2, 4], 'transformer_depth': [1, 1, 1, 0], 'channel_mult': [1, 2, 4, 4],
                    'transformer_depth_middle': 1, 'use_linear_in_transformer': True, 'context_dim': 1024}

    SD15 = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
            'adm_in_channels': None, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320, 'num_res_blocks': 2,
            'attention_resolutions': [1, 2, 4], 'transformer_depth': [1, 1, 1, 0], 'channel_mult': [1, 2, 4, 4],
            'transformer_depth_middle': 1, 'use_linear_in_transformer': False, 'context_dim': 768, "num_heads": 8}

    SDXL_mid_cnet = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
            'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
            'num_res_blocks': 2, 'attention_resolutions': [4], 'transformer_depth': [0, 0, 1], 'channel_mult': [1, 2, 4],
            'transformer_depth_middle': 1, 'use_linear_in_transformer': True, 'context_dim': 2048, "num_head_channels": 64}

    SDXL_small_cnet = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
            'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
            'num_res_blocks': 2, 'attention_resolutions': [], 'transformer_depth': [0, 0, 0], 'channel_mult': [1, 2, 4],
            'transformer_depth_middle': 0, 'use_linear_in_transformer': True, "num_head_channels": 64, 'context_dim': 1}

    SDXL_diffusers_inpaint = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
            'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 9, 'model_channels': 320,
            'num_res_blocks': 2, 'attention_resolutions': [2, 4], 'transformer_depth': [0, 2, 10], 'channel_mult': [1, 2, 4],
            'transformer_depth_middle': 10, 'use_linear_in_transformer': True, 'context_dim': 2048, "num_head_channels": 64}

    supported_models = [SDXL, SDXL_refiner, SD21, SD15, SD21_uncliph, SD21_unclipl, SDXL_mid_cnet, SDXL_small_cnet, SDXL_diffusers_inpaint]

    for unet_config in supported_models:
        matches = True
        for k in match:
            if match[k] != unet_config[k]:
                matches = False
                break
        if matches:
            return unet_config
    return None

def model_config_from_diffusers_unet(state_dict, dtype):
    unet_config = unet_config_from_diffusers_unet(state_dict, dtype)
    if unet_config is not None:
        return model_config_from_unet_config(unet_config)
    return None
