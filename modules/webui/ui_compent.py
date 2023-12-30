import random
import re
import gradio as gr
import modules.options as opts
from modules import lora, wd14tagger
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
    gl_sample_list = gr.Gallery(label="Generated images", show_label=False, columns=[8], rows=[1], object_fit="contain", elem_id="gl_sample_list")
    num_selected_sample = gr.Number(gl_sample_list.selected_index or -1, visible=False, elem_id="num_selected_sample")
    txt_sample_info = gr.Text(visible=False, show_label=False, container=False, lines=5, elem_id="txt_sample_info")
    
    return gl_sample_list, num_selected_sample, txt_sample_info

def StyleGallery():
    gl_style_list = gr.Gallery(value=ui_process.GetStyleList(opts.default['style']), show_label=False, selected_index=0, height=135, columns=[11], rows=[2], show_download_button=False, allow_preview=False, object_fit="cover", elem_id="gl_style_list")
    num_selected_style = gr.Number(gl_style_list.selected_index or 0, visible=False, elem_id="txt_selected_style")
    
    def on_select(evt: gr.SelectData):  # SelectData is a subclass of EventData
        return gr.Gallery(selected_index=evt.index), evt.index

    gl_style_list.select(on_select, None, [gl_style_list, num_selected_style], queue=False)

    return gl_style_list, num_selected_style

def LoraBlock(opt_dict, loraCount=5, showCount=3):
    blocks = []
    ctrls = []
    lora_label = list(opts.options['ref_mode'].keys())[-1]
    btn_loras = []
    opt_loras = []

    def refresh_lora():
        return [ gr.Dropdown(choices=["None"] + list(lora.get_list().keys())) ] * loraCount

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
        with gr.Column(min_width=100, visible=i < showCount) as block:
            image_lora = gr.Image(show_label=False, height=200, show_download_button=False, type="filepath", interactive=False, elem_id=f"refer_lora_img_{i}", elem_classes="lora_preview")
            with gr.Row():
                opt_lora = gr.Dropdown(choices=["None"] + list(lora.lora_files.keys()), value="None", container=False, filterable=False, elem_id=f"refer_lora_{i}", elem_classes="gr_dropdown", scale=95)
                btn_refresh_lora = gr.Button("\U0001f504", min_width=20, scale=5, elem_id=f"refer_lora_refresh_{i}")
            html_link = gr.HTML()
            opt_trained_words = gr.Dropdown(label="Trained Words", visible=False, filterable=False, elem_id=f"refer_trained_words_{i}", elem_classes="gr_dropdown")
            sl_weight = gr.Slider(minimum=0, maximum=200, value=80, step=5, label="Weight", visible=False, elem_id=f"refer_weight_{i}")

            opt_dict[f"refer_lora_{i}"] = opt_lora
            opt_dict[f"refer_lora_trained_words_{i}"] = opt_trained_words
            opt_dict[f"refer_lora_weight_{i}"] = sl_weight
            
        btn_loras.append(btn_refresh_lora)
        opt_loras.append(opt_lora)

        opt_lora.change(select_lora, opt_lora, [image_lora, opt_trained_words, sl_weight, html_link], queue=False)

        blocks.append(block)
        ctrls += [opt_lora, sl_weight, opt_trained_words]

    for btn_lora in btn_loras:
        btn_lora.click(refresh_lora, None, opt_loras, queue=False)

    return blocks, ctrls

def RefBlock(opt_base_model, opt_dict, refCount=5, showCount=3):
    blocks = []
    ctrls = []
    
    def get_tags(opt_type, img_refer):
        words = []
        if img_refer is not None and opt_type == "Ref Content":
            words = wd14tagger.tag(img_refer)[:12]
            
        return gr.CheckboxGroup(choices=words, value=words, visible=len(words) > 0)

    def get_models(opt_type, opt_base_model):
        if opt_type in ["Ref Content", "Base Image"]:
            return gr.Dropdown(visible=False)
        else:
            model_type = "sd15" if opt_base_model and "(sd15)" in opt_base_model else "sdxl"
            ctrl_models, default_model = ui_process.GetControlnets(opt_type, model_type)
            return gr.Dropdown(choices=ctrl_models, value=default_model, visible=len(ctrl_models) > 1)

    ctrl_models, default_model = ui_process.GetControlnets(opts.default["ref_mode"], "sdxl")

    for i in range(refCount):
        with gr.Column(min_width=100, visible=i < showCount) as block:
            image_refer = gr.Image(label=opts.title['ref_image'], height=200, elem_id=f"refer_img_{i}")
            sl_rate = gr.Slider(minimum=5, maximum=100, value=60, step=5, label="Ref Rate", visible=False, elem_id=f"refer_rate_{i}")
            ckb_words = gr.CheckboxGroup(show_label=False, visible=False, elem_id=f"refer_wd14_{i}", elem_classes="refer_words")
            
            opt_type_list = list(opts.options['ref_mode'].keys())
            opt_type_list = [x for x in opt_type_list if i == 0 or x != "Base Image"]
            opt_type = gr.Dropdown(choices=opt_type_list, value=opts.default["ref_mode"], container=False, filterable=False, min_width=80, elem_id=f"refer_type_{i}", elem_classes="gr_dropdown")

            opt_model = gr.Dropdown(choices=ctrl_models, value=default_model, visible=(len(ctrl_models) > 1),  container=False, filterable=False, min_width=80, elem_id=f"ref_model_{i}", elem_classes="gr_dropdown")

        opt_dict[f"refer_type_{i}"] = opt_type

        opt_type.change(get_tags, [opt_type, image_refer], [ckb_words], queue=False) \
            .then(get_models, [opt_type, opt_base_model], opt_model, queue=False)

        opt_base_model.change(get_models, [opt_type, opt_base_model], opt_model, queue=False)

        image_refer.change(lambda x: gr.Slider(visible=x is not None), image_refer, [sl_rate], queue=False) \
            .then(get_tags, [opt_type, image_refer], [ckb_words], queue=False)

        blocks.append(block)
        ctrls += [opt_type, image_refer, sl_rate, ckb_words, opt_model]
    
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

def ReFinerFaceUI(panel_action_btns, panel_action_interface, btn_refiner_face_interface, opt_dict):
    with gr.Row(visible=False) as panel_refiner_face:
        opt_dict["refiner_face_denoise"] = MakeSlider(["refiner_face_denoise"], min_width=160)["refiner_face_denoise"]
        opt_dict["refiner_face_prompt"] = gr.Text(label="Prompt Content", elem_id="txt_refiner_face_prompt", elem_classes="prompt")
        btn_refiner_face = gr.Button("Refiner Face", min_width=100, variant="primary", elem_id="btn_resize", elem_classes="btn_action")
        btn_refiner_face_cancel = gr.Button("Close", min_width=100, elem_id="btn_resize_cancel", elem_classes="btn_action")
    
    btn_refiner_face_interface.click(lambda:(gr.Column(visible=False), gr.Column(visible=True), gr.Row(visible=True)), None, [panel_action_btns, panel_action_interface, panel_refiner_face], queue=False)
    btn_refiner_face_cancel.click(lambda:(gr.Column(visible=True), gr.Column(visible=False), gr.Row(visible=False)), None, [panel_action_btns, panel_action_interface, panel_refiner_face], queue=False)

    return btn_refiner_face

def VaryCustomUI(panel_action_btns, panel_action_interface, btn_vary_custom_interface, opt_dict, gl_sample_list, num_selected_sample, panel_sample_gallery, panel_editor, img_vary_editor):
    with gr.Row(visible=False) as panel_vary_custom:
        opt_dict["vary_custom_strength"] = MakeSlider(["vary_custom_strength"], min_width=160)["vary_custom_strength"]
        opt_dict["vary_custom_area"] = MakeOpts(["vary_custom_area"], opt_type="Radio", min_width=160)["vary_custom_area"]
        opt_dict["vary_prompt"] = gr.Text(label="Prompt Content", elem_id="txt_vary_prompt", elem_classes="prompt")
        btn_vary_custom = gr.Button("Vary", min_width=100, variant="primary", elem_id="btn_vary_custom", elem_classes="btn_action")
        btn_vary_custom_cancel = gr.Button("Close", min_width=100, elem_id="btn_vary_custom_cancel", elem_classes="btn_action")
    
    btn_vary_custom_interface.click(lambda: gr.Image(value=None), None, img_vary_editor, queue=False) \
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

