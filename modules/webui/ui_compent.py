import random
import re
import time
import gradio as gr
import modules.options as opts
from modules import lora, wd14tagger
from modules.image2text import Image2TextProcessor
from modules.webui import ui_process

mc_show_count = 7

def MakeOpts(items, opt_type="Dropdown", show_label=True, container=True, visible=True, min_width=80, scale=None, interactive=True):
    opt_list = {}
    gr_class = {
        "Dropdown": (gr.Dropdown, { "filterable": False }),
        "Radio": (gr.Radio, {}),
        "CheckboxGroup": (gr.CheckboxGroup, {}),
    }

    for item in items:
        opt = gr_class[opt_type][0](
                opts.options[item].keys(),
                value=opts.default[item] if opt_type not in ["Dropdown"] else None,
                label=opts.title[item],
                show_label=show_label,
                container=container,
                visible=visible,
                min_width=min_width,
                scale=scale,
                interactive=interactive,
                elem_id=f"opt_{item}",
                elem_classes=f"gr_{opt_type.lower()}",
                **gr_class[opt_type][1]
            )
        
        opt_list[item] = opt

    return opt_list

def MakeSlider(items, show_label=True, min_width=80, container=True, visible=True, scale=None, interactive=True):
    sl_list = {}

    for item in items:
        sl_start, sl_end, sl_step = opts.options[item]
        sl = gr.Slider(
                sl_start,
                sl_end,
                value=opts.default[item],
                label=opts.title[item],
                step=sl_step,
                show_label=show_label,
                container=container,
                visible=visible,
                min_width=min_width,
                scale=scale,
                interactive=interactive,
                elem_id=f"sl_{item}",
                elem_classes="gr_slider",
            )

        sl_list[item] = sl

    return sl_list

def MC(show_count):
    global mc_show_count
    mc_show_count = show_count

    default_item = opts.default["mc"]

    with gr.Row() as panel_mc:
        radio_mc = gr.Radio(choices=get_mc(default_item), value=default_item, label=opts.title['mc'], show_label=True, container=True, elem_classes="label-left", elem_id="radio_mc", scale=99)
        btn_refresh_mc = gr.Button("Refresh MC", size="sm", min_width=100, elem_classes="label-left", elem_id="btn_refresh_mc", scale=1)
    with gr.Row(visible=False) as panel_mc_other:
        ckb_mc_other = gr.CheckboxGroup(choices=["Other"], value=["Other"], label=opts.title['mc'], elem_classes="label-left", scale=10, min_width=250, elem_id="ckb_mc_other")
        txt_mc_other = gr.Textbox(show_label=False, container=False, scale=20, elem_id="txt_mc_other", elem_classes="prompt")
        gr.CheckboxGroup(scale=70, container=False)

    btn_refresh_mc.click(lambda x : gr.Radio(choices=get_mc(x)), [radio_mc], [radio_mc], queue=False)
    radio_mc.change(lambda x: (gr.Row(visible=not x=='Other'), gr.Row(visible=x=='Other')), radio_mc, [panel_mc, panel_mc_other], queue=False)
    ckb_mc_other.change(lambda x: (gr.Row(visible=True), gr.Row(visible=False), gr.Radio(value=opts.default['mc']), gr.CheckboxGroup(value=['Other'])), ckb_mc_other, [panel_mc, panel_mc_other, radio_mc, ckb_mc_other], queue=False)

    return radio_mc, ckb_mc_other, txt_mc_other

def get_mc(select_item):
    items = list(filter(lambda x: x not in [select_item, "Other"], opts.options["mc"].keys()))
    random.shuffle(items)
    choices = [select_item] + items[:mc_show_count] + ["Other"]
    return choices

def SetMCShowCount(opt_lang, radio_mc):
    global mc_show_count
    mc_show_count = 9 if opt_lang in ["zh-CN", "zh-TW"] else 7
    return gr.Radio(choices=get_mc(radio_mc))

def SampleGallery():
    # img_preview = gr.Image(label='Preview', show_label=True, visible=False, elem_id="img_preview", scale=20)
    # with gr.Column(scale=85):
    gl_sample_list = gr.Gallery(label="Generated images", show_label=False, columns=[8], rows=[1], object_fit="contain", interactive=False, allow_preview=False, elem_id="gl_sample_list", elem_classes="image_gallery")
    num_selected_sample = gr.Number(gl_sample_list.selected_index or -1, visible=False, elem_id="num_selected_sample")
    txt_sample_info = gr.Text(visible=False, show_label=False, container=False, lines=5, elem_id="txt_sample_info")
    
    return gl_sample_list, num_selected_sample, txt_sample_info

def StyleGallery():
    gl_style_list = gr.Gallery(value=ui_process.GetStyleList(opts.default['style']), show_label=False, selected_index=0, interactive=False, height=135, columns=[10], rows=[2], show_download_button=False, allow_preview=False, object_fit="cover", elem_id="gl_style_list")
    num_selected_style = gr.Number(gl_style_list.selected_index or 0, visible=False, elem_id="txt_selected_style")
    
    def on_select(evt: gr.SelectData):  # SelectData is a subclass of EventData
        return gr.Gallery(selected_index=evt.index), evt.index

    gl_style_list.select(on_select, None, [gl_style_list, num_selected_style], queue=False)

    return gl_style_list, num_selected_style

def LoraBlock(opt_dict, loraCount=6, showCount=3):
    blocks = []
    ctrls = []
    lora_label = list(opts.options['ref_mode'].keys())[-1]
    btn_loras = []
    opt_loras = []

    def select_lora(opt_lora):
        trained_words = []
        prview = None
        html = None

        if opt_lora is not None and opt_lora != "None":
            info = lora.get_info(opt_lora)
            if info is not None:
                trained_words = [x.strip(' ,') for x in info.get("trainedWords", [])]
                prview = info["preview_image"] or info["images"][0]["url"]
                url = info["model_homepage"]
                base_model = info["baseModel"]
                html = f"<span>Based on</span> [ <span>{base_model}</span> ] <span> </span> <a href='{url}' target='_blank'>Model details</a>"

        default_words = trained_words[0] if len(trained_words) > 0 else None

        return  gr.Image(prview), \
                gr.Dropdown(choices=trained_words, value=default_words, visible=len(trained_words) > 0), \
                gr.Slider(visible=(opt_lora is not None and opt_lora != "None")), \
                gr.HTML(html)

    for i in range(loraCount):
        with gr.Column(min_width=200, visible=i < showCount, elem_classes="panel_lora_block") as block:
            image_lora = gr.Image(show_label=False, height=200, show_download_button=False, type="filepath", interactive=False, elem_id=f"refer_lora_img_{i}", elem_classes="lora_preview")
            with gr.Row():
                ckb_enable = gr.Checkbox(label="", value=True, min_width=40, elem_id=f"refer_lora_enable_{i}", scale=1)
                opt_lora = gr.Dropdown(choices=["None"] + list(lora.lora_files.keys()), value="None", container=False, filterable=True, elem_id=f"refer_lora_{i}", elem_classes="gr_dropdown", scale=95)
            html_link = gr.HTML()
            opt_trained_words = gr.Dropdown(label="Trained Words", visible=False, filterable=False, elem_id=f"refer_trained_words_{i}", elem_classes="gr_dropdown")
            sl_weight = gr.Slider(minimum=0, maximum=200, value=80, step=5, label="Weight", visible=False, elem_id=f"refer_weight_{i}")

            opt_dict[f"refer_lora_{i}"] = opt_lora
            opt_dict[f"refer_lora_trained_words_{i}"] = opt_trained_words
            opt_dict[f"refer_lora_weight_{i}"] = sl_weight
            
        opt_loras.append(opt_lora)

        opt_lora.change(select_lora, opt_lora, [image_lora, opt_trained_words, sl_weight, html_link], queue=False)
        ckb_enable.change(lambda x: gr.Dropdown(interactive=x), ckb_enable, opt_lora, queue=False)

        blocks.append(block)
        ctrls += [opt_lora, ckb_enable, sl_weight, opt_trained_words]

    return blocks, ctrls

def RefBlock(opt_base_model, opt_dict, refCount=5, showCount=3):
    blocks = []
    ctrls = []
    
    def image_refer_change(image_refer):            
        return  (   gr.Slider(visible=image_refer is not None), # sl_rate
                    gr.Slider(visible=image_refer is not None), # sl_start_percent
                    gr.Slider(visible=image_refer is not None), # sl_end_percent
                    gr.Column(visible=image_refer is not None)) # panel_mask

    def opt_type_change(opt_type, opt_image2text, image_refer):
        image2text_visible = str(opt_type) in ["Ref Content"] and image_refer is not None
        range_visible = str(opt_type) not in ["Ref Content", "Base Image"]
        wd14_visible = image2text_visible and "wd14" in opt_image2text
        return gr.Column(visible=image2text_visible), gr.Row(visible=range_visible), gr.CheckboxGroup(visible=wd14_visible)
    
    def get_tags(opt_type, opt_image2text, img_refer):
        words, text = [], ""

        if img_refer is not None and opt_type == "Ref Content":
            process_name = image2texts.get(opt_image2text)
            process = Image2TextProcessor(process_name)
            if "wd14" in opt_image2text.lower():
                words = wd14tagger.tag(img_refer)[:15]
                text = ""
            else:
                words = []
                text = process(img_refer, max_new_tokens=256)
        
        panel_visible = True if str(opt_type) in ["Ref Content"] else False
        words_visible = True if words else False
        text_visible = True if text else False
            
        return gr.Column(visible=panel_visible), gr.CheckboxGroup(choices=words, value=words, visible=words_visible), gr.Textbox(text, visible=text_visible)

    def get_ref_types_inner(opt_base_model):
        # opt_type_list = list(opts.options['ref_mode'].keys())
        model_type = "sd15" if opt_base_model and "(sd15)" in str(opt_base_model) else "sdxl"
        opt_type_list = [k for k, v in opts.options['ref_mode'].items() \
            if k in ["Ref Content", "Base Image"] or v[1].get(model_type) is not None]

        return opt_type_list

    def get_ref_types(opt_base_model):
        ref_list = get_ref_types_inner(opt_base_model)

        return gr.Dropdown(choices=ref_list)

    def get_models(opt_type, opt_base_model, opt_ctrl_model):
        if opt_type in ["Ref Content", "Base Image"]:
            return gr.Dropdown(visible=False)
        else:
            model_type = "sd15" if opt_base_model and "(sd15)" in opt_base_model else "sdxl"
            ctrl_models, default_model = ui_process.GetControlnets(opt_type, model_type)
            if opt_ctrl_model in ctrl_models:
                return gr.Dropdown(choices=ctrl_models, visible=len(ctrl_models) > 1)
            else:
                return gr.Dropdown(choices=ctrl_models, value=default_model, visible=len(ctrl_models) > 1)
    
    def get_annotators_inner(opt_type, opt_ctrl_model):
        if opt_type is None:
            return None, None
        opt_config = opts.options["ref_mode"][opt_type][1]
        keyword = opt_config.get("keyword", [None])
        keyword = keyword if isinstance(keyword, list) else [keyword]
        opt_ctrl_model = re.sub(r"[\-_]", " ", opt_ctrl_model or "").lower()
        model_annotators = opt_config.get("annotator")
        model_annotators = model_annotators if isinstance(model_annotators, list) else [model_annotators]
        model_annotators = [a if isinstance(a, list) else [a] for k, a in zip(keyword, model_annotators) if re.search(k, opt_ctrl_model)]

        model_annotators = model_annotators[0] if model_annotators else []
        default_annotator = model_annotators[0] if model_annotators else None

        return model_annotators, default_annotator
    
    def get_annotators(opt_type, opt_model, opt_annotator):
        if opt_type in ["Ref Content", "Base Image"]:
            return gr.Dropdown(), gr.Row(visible=False)
        else:
            ctrl_annotators, default_annotator = get_annotators_inner(opt_type, opt_model)
            visible = len(ctrl_annotators) > 0 and ctrl_annotators != ["ip_adapter"] and ctrl_annotators != ["ip_adapter_face"]
            if opt_annotator in ctrl_annotators:
                return gr.Dropdown(choices=ctrl_annotators), gr.Row(visible=visible)
            else:
                return gr.Dropdown(choices=ctrl_annotators, value=default_annotator), gr.Row(visible=visible)

    default_ref_mode = opts.default["ref_mode"]
    ctrl_models, default_model = ui_process.GetControlnets(default_ref_mode, "sdxl")
    ctrl_annotators, default_annotator = get_annotators_inner(default_ref_mode, default_model)
    image2texts, default_image2text = {
            "Wd14 Tagger v3": "wd14_v3",
            "Moondream v1": "moondream_v1",
            "Moondream v2": "moondream_v2",
            "QianWen": "qwen",
        }, "Wd14 Tagger v3"

    for i in range(refCount):
        with gr.Column(min_width=100, visible=i < showCount, elem_classes="panel_ref_block") as block:
            image_refer = gr.Image(label=opts.title['ref_image'], height=280, elem_id=f"refer_img_{i}")
            
            opt_type_list = get_ref_types_inner(opt_base_model)
            # opt_type_list = [x for x in opt_type_list if i == 0 or x != "Base Image"]
            with gr.Row():
                ckb_enable = gr.Checkbox(label="", value=True, min_width=40, elem_id=f"refer_enable_{i}", scale=1)
                opt_type = gr.Dropdown(choices=opt_type_list, value=opts.default["ref_mode"], container=False, filterable=False, min_width=80, elem_id=f"refer_type_{i}", elem_classes="gr_dropdown refer_type", scale=9)

            sl_rate = gr.Slider(minimum=5, maximum=150, value=60, step=5, label="Ref Rate %", visible=False, elem_id=f"refer_rate_{i}")
            with gr.Row() as panel_step_range:
                sl_start_percent = gr.Slider(minimum=0, maximum=100, value=0, step=5, min_width=120, label="Start At %", visible=False, elem_id=f"refer_start_{i}", elem_classes="refer_start_at")
                sl_end_percent = gr.Slider(minimum=0, maximum=100, value=80, step=5, min_width=120, label="End At %", visible=False, elem_id=f"refer_end_{i}", elem_classes="refer_end_at")
                with gr.Column(visible=False) as panel_mask:
                    ckb_mask = gr.Checkbox(label="apply mask", value=False, min_width=40, elem_id=f"refer_apply_mask_{i}")
                    image_attn_mask = gr.Image(label="mask", height=280, visible=False, elem_id=f"refer_mask_{i}")

            with gr.Column(visible=False) as panel_image2text:
                opt_image2text = gr.Dropdown(choices=image2texts.keys(), value=default_image2text, container=False, filterable=False, min_width=80, elem_id=f"ref_image2text_{i}", elem_classes="gr_dropdown")
                ckb_words = gr.CheckboxGroup(show_label=False, label="", visible=True, elem_id=f"refer_wd14_{i}", elem_classes="refer_words")
                txt_words = gr.Textbox(show_label=False, visible=False, elem_id=f"refer_img2text_{i}")
                
            opt_ctrl_model = gr.Dropdown(choices=ctrl_models, value=default_model, visible=(len(ctrl_models) > 1),  container=False, filterable=False, min_width=80, elem_id=f"ref_ctrl_model_{i}", elem_classes="gr_dropdown")
            with gr.Row(visible=(len(ctrl_annotators) > 1)) as panel_annotator:
                opt_annotator = gr.Dropdown(choices=ctrl_annotators, value=default_annotator, container=False, filterable=False, min_width=80, elem_id=f"ref_annotator_{i}", elem_classes="gr_dropdown", scale=9)
                ckb_annotator = gr.Checkbox(show_label=False, label="", value=True, min_width=40, elem_id=f"refer_use_annotator_{i}", scale=1)

        opt_dict[f"refer_type_{i}"] = opt_type

        opt_type.change(opt_type_change, [opt_type, opt_image2text, image_refer], [panel_image2text, panel_step_range, ckb_words], queue=False) \
            .then(get_models, [opt_type, opt_base_model, opt_ctrl_model], [opt_ctrl_model], queue=False) \
            .then(get_annotators, [opt_type, opt_ctrl_model, opt_annotator], [opt_annotator, panel_annotator], queue=False) \
            .then(get_tags, [opt_type, opt_image2text, image_refer], [panel_image2text, ckb_words, txt_words], queue=False) \

        opt_image2text.change(get_tags, [opt_type, opt_image2text, image_refer], [panel_image2text, ckb_words, txt_words], queue=False)

        opt_base_model.change(get_ref_types, opt_base_model, opt_type, queue=False) \
            .then(get_models, [opt_type, opt_base_model, opt_ctrl_model], opt_ctrl_model, queue=False)
        opt_ctrl_model.change(get_annotators, [opt_type, opt_ctrl_model, opt_annotator], [opt_annotator, panel_annotator], queue=False)

        image_refer.change(image_refer_change, image_refer, [sl_rate, sl_start_percent, sl_end_percent, panel_mask], queue=False) \
            .then(opt_type_change, [opt_type, opt_image2text, image_refer], [panel_image2text, panel_step_range, ckb_words], queue=False) \
            .then(get_tags, [opt_type, opt_image2text, image_refer], [panel_image2text, ckb_words, txt_words], queue=False) \

        ckb_enable.change(lambda x: gr.Dropdown(interactive=x), ckb_enable, opt_type, queue=False)
        ckb_annotator.change(lambda x: gr.Dropdown(interactive=x), ckb_annotator, opt_annotator, queue=False)

        ckb_mask.change(lambda x: gr.Image(visible=x), ckb_mask, image_attn_mask, queue=False)

        blocks.append(block)
        ctrls += [opt_type, ckb_enable, image_refer, sl_rate, ckb_words, txt_words, opt_ctrl_model, ckb_annotator, opt_annotator, ckb_mask, image_attn_mask, sl_start_percent, sl_end_percent]
    
    return blocks, ctrls

def ReFaceUI(panel_action_btns, panel_action_interface, btn_reface_interface, opt_dict):
    with gr.Row(visible=False) as panel_reface:
        img_face = gr.Image(label="ReFace", show_label=False, width=200, height=160, min_width=200, elem_id="img_face")
        btn_reface = gr.Button("ReFace", min_width=100, variant="primary", interactive=False, elem_id="btn_reface", elem_classes="btn_action")
        btn_reface_cancel = gr.Button("Close", min_width=100, elem_id="btn_reface_cancel", elem_classes="btn_action")

    img_face.change(lambda x: gr.Button(interactive=x is not None), img_face, btn_reface)
    btn_reface_interface.click(lambda:(gr.Column(visible=False), gr.Column(visible=True), gr.Row(visible=True)), None, [panel_action_btns, panel_action_interface, panel_reface], queue=False)
    btn_reface_cancel.click(lambda:(gr.Column(visible=True), gr.Column(visible=False), gr.Row(visible=False)), None, [panel_action_btns, panel_action_interface, panel_reface], queue=False)

    return btn_reface, img_face

def ReFinerUI(panel_action_btns, panel_action_interface, btn_refiner_interface, opt_dict, type="image"):
    action_name = "Refiner Image" if type == "image" else "Refiner Face"
    with gr.Row(visible=False) as panel_refiner:
        opt_dict[f"refiner_{type}_denoise"] = MakeSlider(["refiner_denoise"], min_width=160)["refiner_denoise"]
        opt_dict[f"refiner_{type}_detail"] = MakeSlider(["refiner_detail"], min_width=160)["refiner_detail"]
        if type == "face":
            opt_dict["refiner_face_index"] = MakeOpts(["refiner_face_index"], opt_type="Radio", min_width=160)["refiner_face_index"]
        opt_dict[f"refiner_{type}_prompt"] = gr.Text(label="Prompt Content", elem_id="txt_refiner_prompt", elem_classes="prompt")
        btn_refiner = gr.Button(action_name, min_width=100, variant="primary", elem_id="btn_refiner", elem_classes="btn_action")
        btn_refiner_cancel = gr.Button("Close", min_width=100, elem_id="btn_refiner_cancel", elem_classes="btn_action")
    
    btn_refiner_interface.click(lambda:(gr.Column(visible=False), gr.Column(visible=True), gr.Row(visible=True)), None, [panel_action_btns, panel_action_interface, panel_refiner], queue=False)
    btn_refiner_cancel.click(lambda:(gr.Column(visible=True), gr.Column(visible=False), gr.Row(visible=False)), None, [panel_action_btns, panel_action_interface, panel_refiner], queue=False)

    return btn_refiner

def VaryCustomUI(panel_action_btns, panel_action_interface, btn_vary_custom_interface, opt_dict, gl_sample_list, num_selected_sample, panel_sample_gallery, panel_editor, img_vary_editor):
    with gr.Row(visible=False) as panel_vary_custom:
        opt_dict["vary_custom_strength"] = MakeSlider(["vary_custom_strength"], min_width=160)["vary_custom_strength"]
        opt_dict["vary_custom_area"] = MakeOpts(["vary_custom_area"], opt_type="Radio", min_width=160)["vary_custom_area"]
        ckb_mask_lama = gr.Checkbox(label="Vary after Erase", value=False, elem_id="ckb_mask_lama")      
        opt_dict["mask_use_lama"] = ckb_mask_lama
        opt_dict["vary_prompt"] = gr.Text(label="Prompt Content", elem_id="txt_vary_prompt", elem_classes="prompt")
        btn_vary_custom = gr.Button("Vary", min_width=100, variant="primary", elem_id="btn_vary_custom", elem_classes="btn_action")
        btn_vary_custom_cancel = gr.Button("Close", min_width=100, elem_id="btn_vary_custom_cancel", elem_classes="btn_action")
    
    btn_vary_custom_interface.click(ui_process.VaryClearMask, [num_selected_sample], img_vary_editor, queue=False) \
        .then(ui_process.VaryCustomInterface, [gl_sample_list, num_selected_sample], [panel_action_btns, panel_action_interface, panel_vary_custom, panel_sample_gallery, panel_editor, img_vary_editor], queue=False)
    btn_vary_custom.click(lambda:(gr.Column(visible=True), gr.Column(visible=False), gr.Row(visible=False), gr.Column(visible=True), gr.Column(visible=False)), None, [panel_action_btns, panel_action_interface, panel_vary_custom, panel_sample_gallery, panel_editor], queue=False)
    btn_vary_custom_cancel.click(lambda:(gr.Column(visible=True), gr.Column(visible=False), gr.Row(visible=False), gr.Column(visible=True), gr.Column(visible=False)), None, [panel_action_btns, panel_action_interface, panel_vary_custom, panel_sample_gallery, panel_editor], queue=False)

    return btn_vary_custom

def ZoomCustomUI(panel_action_btns, panel_action_interface, btn_zoom_custom_interface, btn_resize_interface, opt_dict):
    with gr.Row(visible=False) as panel_zoom_custom:
        sl_zoom_custom = opt_dict["zoom_custom"] = MakeSlider(["zoom_custom"], min_width=160)["zoom_custom"]
        opt_resize_ratios = opt_dict["resize_ratios"] = MakeOpts(["resize_ratios"], min_width=160, visible=False)["resize_ratios"]
        opt_dict["zoom_denoise"] = MakeSlider(["zoom_denoise"], min_width=160, visible=False)["zoom_denoise"]
        opt_dict["zoom_blur_alpha"] = MakeSlider(["zoom_blur_alpha"], min_width=160)["zoom_blur_alpha"]
        opt_dict["zoom_prompt"] = gr.Text(label="Prompt Content", elem_id="txt_zoom_prompt", elem_classes="prompt")
        btn_zoom_custom = gr.Button("Zoom", min_width=100, variant="primary", elem_id="btn_zoom", elem_classes="btn_action")
        btn_resize = gr.Button("ReSize", min_width=100, variant="primary", elem_id="btn_resize", elem_classes="btn_action")
        btn_zoom_cancel = gr.Button("Close", min_width=100, elem_id="btn_zoom_cancel", elem_classes="btn_action")

    btn_zoom_custom_interface.click(lambda:(gr.Column(visible=False), gr.Column(visible=True), gr.Row(visible=True), gr.Button(visible=True), gr.Button(visible=False), gr.Slider(visible=True), gr.Dropdown(visible=False)), None, [panel_action_btns, panel_action_interface, panel_zoom_custom, btn_zoom_custom, btn_resize, sl_zoom_custom, opt_resize_ratios], queue=False)
    btn_resize_interface.click(lambda:(gr.Column(visible=False), gr.Column(visible=True), gr.Row(visible=True), gr.Button(visible=False), gr.Button(visible=True), gr.Slider(visible=False), gr.Dropdown(visible=True)), None, [panel_action_btns, panel_action_interface, panel_zoom_custom, btn_zoom_custom, btn_resize, sl_zoom_custom, opt_resize_ratios], queue=False)
    btn_zoom_cancel.click(lambda:(gr.Column(visible=True), gr.Column(visible=False), gr.Row(visible=False)), None, [panel_action_btns, panel_action_interface, panel_zoom_custom], queue=False)

    return btn_zoom_custom, btn_resize

def UpscaleUI(panel_action_btns, panel_action_interface, btn_interface, opt_dict):
    with gr.Row(visible=False) as panel_ui:
        opt_dict["upscale_factor"] = MakeSlider(["upscale_factor"], min_width=160)["upscale_factor"]
        opt_dict["upscale_model"] = MakeOpts(["upscale_model"], min_width=160)["upscale_model"]
        opt_dict["upscale_origin"] = MakeSlider(["upscale_origin"], min_width=160)["upscale_origin"]
        ckb_upscale_repair_face = gr.Checkbox(label="Repair Face", elem_id="ckb_upscale_repair_face", value=True)
        opt_dict["upscale_repair_face"] = ckb_upscale_repair_face

        btn_upscale = gr.Button("Upscale", min_width=100, elem_id="btn_upscale", variant="primary", elem_classes="btn_action")
        btn_cancel = gr.Button("Close", min_width=100, elem_id="btn_upscale_cancel", elem_classes="btn_action")

    btn_interface.click(lambda:(gr.Column(visible=False), gr.Column(visible=True), gr.Row(visible=True)), None, [panel_action_btns, panel_action_interface, panel_ui], queue=False)
    btn_cancel.click(lambda:(gr.Column(visible=True), gr.Column(visible=False), gr.Row(visible=False)), None, [panel_action_btns, panel_action_interface, panel_ui], queue=False)

    return btn_upscale

