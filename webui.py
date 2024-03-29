import json
import gradio as gr
from modules import localization, shared
from modules.model import model_helper
from modules.webui import ui_action, ui_compent, ui_extensions, ui_process

import modules.options as opts

my_theme = gr.themes.Soft()
app_name = "Wooool~"

opt_dict = {}
opt_weights = {}
ref_image_count = ui_process.ref_image_count

ui_extensions.render_ui()

with gr.Blocks(theme=my_theme, title=app_name).queue() as wooool:
    opt_lang = gr.Dropdown(choices=localization.urls().keys(), value="en", container=False, filterable=False, min_width=50, elem_id=f"opt_lang", elem_classes="gr_dropdown")
    with gr.Column(visible=False) as main_ui:
        with gr.Column(elem_id="panel_header") as panel_header:
            title = gr.Button(f"{app_name}", visible=True, elem_id="title")
            
            with gr.Row(elem_id="panel_gallery", visible=False) as panel_gallery:
                with gr.Column(scale=85, visible=True) as panel_sample_gallery:
                    gl_sample_list, num_selected_sample, txt_sample_info = ui_compent.SampleGallery()
                with gr.Column(scale=85, visible=False) as panel_editor:
                    img_vary_editor = gr.Image(label="Vary(Custom)", source="upload", tool="sketch", interactive=True, height=444, brush_radius=80, brush_color="#4338CA", elem_id="img_very_editor", elem_classes="inpaint_canvas")

                with gr.Column(elem_id="panel_action_menu", scale=15, min_width=280) as panel_action_menu:
                    with gr.Row():
                        gr.CheckboxGroup(scale=3, container=False)
                        btn_close_gallery = gr.Button("Close", size="sm", min_width=80, elem_id="btn_close_gallery", scale=1)

                    with gr.Column(elem_id="panel_action") as panel_action:
                        with gr.Tab("Action"):
                            with gr.Column() as panel_action_btns:
                                with gr.Row():
                                    btn_vary_subtle = gr.Button("Vary(Subtle)", min_width=100, elem_id="btn_vary_subtle", variant="primary", elem_classes="btn_action")
                                    btn_vary_strong = gr.Button("Vary(Strong)", min_width=100, elem_id="btn_vary_strong", variant="primary", elem_classes="btn_action")
                                    btn_vary_custom_interface = gr.Button("Vary(Custom)", min_width=200, elem_id="btn_vary_custom_interface", variant="primary", elem_classes="btn_action")

                                    btn_zoom_out15 = gr.Button("Zoom(1.5x)", min_width=100, elem_id="btn_zoom_out15", variant="primary", elem_classes="btn_action", visible=False)
                                    btn_zoom_out20 = gr.Button("Zoom(2.0x)", min_width=100, elem_id="btn_zoom_out20", variant="primary", elem_classes="btn_action", visible=False)
                                    btn_zoom_custom_interface = gr.Button("Zoom", min_width=100, elem_id="btn_zoom_custom_interface", variant="primary", elem_classes="btn_action")
                                    btn_resize_interface = gr.Button("ReSize", min_width=100, elem_id="btn_resize_interface", variant="primary", elem_classes="btn_action")

                                    btn_change_style = gr.Button("Change Style", min_width=100, elem_id="btn_change_style", variant="primary", elem_classes="btn_action")
                                    btn_refiner_interface = gr.Button("Refiner Image", min_width=100, elem_id="btn_refiner", variant="primary", elem_classes="btn_action")
                                    btn_refiner_face_interface = gr.Button("Refiner Face", min_width=100, elem_id="btn_refiner_face", variant="primary", elem_classes="btn_action")
                                    btn_reface_interface = gr.Button("ReFace", min_width=100, elem_id="btn_reface_interface", variant="primary", elem_classes="btn_action")

                                    btn_upscale_interface = gr.Button("Upscale", min_width=100, elem_id="btn_upscale", variant="primary", elem_classes="btn_action")
                                    # gr.Button("Change BG", min_width=100, variant="primary", elem_classes="btn_action"
                                
                                with gr.Row():
                                    btn_delete = gr.Button("Delete", min_width=100, size="sm", elem_id="btn_delete", elem_classes="btn_action")
                                    btn_top = gr.Button("Set to top", min_width=100, size="sm", elem_id="btn_top", elem_classes="btn_action")

                                with gr.Row():
                                    btn_first = gr.Button("First page", min_width=60, size="sm", elem_id="btn_prev", elem_classes="btn_action")
                                    btn_prev = gr.Button("Prev page", min_width=60, size="sm", elem_id="btn_prev", elem_classes="btn_action")
                                    btn_next = gr.Button("Next page", min_width=60, size="sm", elem_id="btn_next", elem_classes="btn_action")
                                    num_page_sample = gr.Number(step=1, minimum=0, min_width=100, visible=False, elem_id="btn_next")

                        with gr.Tab("Setting"):
                            opt_vary_model = gr.Dropdown(choices=ui_process.GetRefinerModels("sdxl"), value=model_helper.default_base_name, label="Vary & Zoom Model", filterable=False, elem_id=f"opt_vary_model", elem_classes="gr_dropdown")
                            opt_dict = opt_dict | ui_compent.MakeSlider(["sample_pagesize"])
                            sl_sample_pagesize = opt_dict["sample_pagesize"]
                            ckb_hide_history = gr.Checkbox(label="Hide history during generate", elem_id="ckb_hide_history", value=False)
                            opt_dict["hide_history"] = ckb_hide_history

                    with gr.Column(elem_id="panel_action_interface", visible=False) as panel_action_interface:
                        btn_reface, img_face = ui_compent.ReFaceUI(panel_action, panel_action_interface, btn_reface_interface, opt_dict)
                        btn_vary_custom = ui_compent.VaryCustomUI(panel_action, panel_action_interface, btn_vary_custom_interface, opt_dict, gl_sample_list, num_selected_sample, panel_sample_gallery, panel_editor, img_vary_editor)
                        btn_zoom_custom, btn_resize = ui_compent.ZoomCustomUI(panel_action, panel_action_interface, btn_zoom_custom_interface, btn_resize_interface, opt_dict)
                        btn_refiner = ui_compent.ReFinerUI(panel_action, panel_action_interface, btn_refiner_interface, opt_dict, type="image")
                        btn_refiner_face = ui_compent.ReFinerUI(panel_action, panel_action_interface, btn_refiner_face_interface, opt_dict, type="face")
                        btn_upscale = ui_compent.UpscaleUI(panel_action, panel_action_interface, btn_upscale_interface, opt_dict)
        
        with gr.Column(elem_id="panel_main") as panel_main:
            with gr.Row(elem_id="panel_prompt") as panel_prompt:
                with gr.Column(scale=99):
                    radio_mc, ckb_mc_other, txt_mc_other = ui_compent.MC(7)
                    txt_prompt_main = gr.Textbox(lines=2, show_label=False, autofocus=True, container=False, placeholder=opts.title['prompt_placeholder'], elem_id="txt_prompt_main", elem_classes="prompt")
                    opt_dict = opt_dict | {
                        "mc": radio_mc,
                        "mc_other": txt_mc_other,
                        "prompt_main": txt_prompt_main,
                    }
                with gr.Column(scale=1, min_width=130) as panel_refer:
                    img_refer = gr.Image(label=opts.title['ref_image'], show_label=True, width=120, min_width=120, height=100, elem_id="img_refer")
            
            with gr.Row(visible=False) as panel_process:
                with gr.Column(scale=90):
                    html_proccbar = gr.HTML("", elem_id="progress")
                ckb_endless_mode = gr.Checkbox(label="Endless", value=False, visible=False, min_width=120, scale=5)
                btn_skip = gr.Button("Skip", visible=True, variant="stop", size="sm", min_width=80, elem_id="btn_skip", scale=5)
                btn_stop = gr.Button("Stop", visible=True, variant="stop", size="sm", min_width=80, elem_id="btn_stop", scale=5)

            with gr.Row() as panel_generate:
                opt_dict = opt_dict | ui_compent.MakeOpts(['pic_num'], show_label=False, container=False, scale=10)
                btn_generate_one = gr.Button(f"Try One", elem_id="btn_generate_one", min_width=100, scale=5)
                btn_generate = gr.Button(f"Generate x {opts.default['pic_num']}", variant="primary", elem_id="btn_generate", scale=80)

            with gr.Column():
                with gr.Row():
                    opt_dict = opt_dict | ui_compent.MakeOpts(["style", "ratios", "view", "emo", "location", "weather", "hue"])
                
                opt_style = opt_dict["style"]
                opt_ratios = opt_dict["ratios"]
                gl_style_list, num_selected_style = ui_compent.StyleGallery()
                opt_dict = opt_dict | {
                    "style_index": num_selected_style
                }

            with gr.Column(visible=False) as panel_pro:
                with gr.Tab("General"):
                    with gr.Row(elem_id="panel_pro_general") as panel_pro_general:
                        with gr.Column(scale=20, min_width=200):
                            base_models = list(model_helper.base_files)
                            opt_base_model = gr.Dropdown(choices=base_models, label="Draw Model", filterable=False, elem_id=f"opt_base_model", elem_classes="gr_dropdown") # value=model_helper.default_base_name,
                            # html_base_model_info = gr.HTML(show_progress="hidden")
                            btn_base_model_info = gr.Button("Model details", link="#", size="sm", elem_id="btn_base_model_info")
                            opt_dict = opt_dict | {
                                "base_model": opt_base_model,
                            }   | ui_compent.MakeSlider(["ref_num"]) \
                                | ui_compent.MakeOpts(["quality"])
                            sl_refNum = opt_dict["ref_num"]

                        with gr.Column(scale=80):
                            with gr.Row():
                                ref_blocks, ref_list = ui_compent.RefBlock(opt_base_model, opt_dict, *ui_process.ref_image_count)
                            with gr.Row():
                                gr.CheckboxGroup(container=False)
                                gr.CheckboxGroup(container=False)
                with gr.Tab("Fine tune"):
                    with gr.Row():
                        with gr.Column(scale=60):
                            with gr.Row():
                                lora_blocks, lora_list = ui_compent.LoraBlock(opt_dict, *ui_process.lora_count)
                            with gr.Row():
                                opt_dict = opt_dict | ui_compent.MakeOpts(["lora_num"], show_label=False, container=False, scale=20)
                                opt_lora_num = opt_dict["lora_num"]
                                btn_prompt_lora = gr.Button("Parse by Prompt", elem_id="btn_prompt_lora", scale=90)
                                btn_refresh_lora = gr.Button("\U0001f504", min_width=20, elem_id=f"btn_lora_refresh", scale=1)
                        with gr.Column(scale=20, min_width=200):
                            opt_weights = opt_weights | ui_compent.MakeSlider(["mc_weight", "style_weight", "view_weight", "emo_weight", "location_weight"])
                        with gr.Column(scale=20, min_width=200):
                            opt_weights = opt_weights | ui_compent.MakeSlider(["weather_weight", "hue_weight", "detail", "more_art_weight"])
                            opt_dict = opt_dict | opt_weights
                            btn_weight_reset = gr.Button("Weight Reset", elem_id="btn_weight_reset")
                with gr.Tab("Advanced"):
                    with gr.Row(elem_id="panel_pro_advanced") as panel_pro_advanced:
                        with gr.Column(scale=20, min_width=200):
                            opt_dict = opt_dict | ui_compent.MakeOpts(["sampler", "scheduler"])
                            opt_dict = opt_dict | ui_compent.MakeSlider(["cfg_scale", "cfg_scale_to", "cfg_multiplier", "free_u", "eta", "clip_skip"])
                        with gr.Column(scale=20, min_width=200):
                            with gr.Row():
                                txt_seed = gr.Number(label="Seed", value=0, elem_id="txt_seed", interactive=False, min_width=80, scale=80)
                                ckb_seed = gr.Checkbox(label="Fixed", value=False, elem_id="ckb_seed", min_width=80, scale=20)
                            opt_dict = opt_dict | ui_compent.MakeSlider(["subseed_strength"], visible=False)
                            sl_subseed_strength = opt_dict["subseed_strength"]
                            
                            opt_refiner_model = gr.Dropdown(choices=[opts.title["disable_refiner"]] + ui_process.GetRefinerModels("sdxl"), value=model_helper.default_refiner_name or opts.title["disable_refiner"], label="Refiner Model", filterable=False, elem_id=f"opt_refiner_model", elem_classes="gr_dropdown")
                            btn_refresh_model = gr.Button("🔄 Refresh Models", elem_id="btn_refresh_model")
                            opt_dict = opt_dict | ui_compent.MakeSlider(["denoise"])
                            opt_dict = opt_dict | ui_compent.MakeSlider(["style_aligned"])
                            opt_dict = opt_dict | ui_compent.MakeOpts(["transparent_bg"], opt_type="Radio")

                            btn_refresh_model.click(ui_process.RefreshModels, None, [opt_base_model, opt_vary_model, opt_refiner_model])
                            # opt_dict = opt_dict | ui_compent.MakeOpts(["single_vae"], opt_type="Radio")
                        with gr.Column(scale=20, min_width=200):
                            txt_prompt_negative = gr.Textbox(label="Negative Prompt", lines=3, placeholder="Input negative.", elem_id="txt_prompt_negative", elem_classes="prompt")
                            opt_dict = opt_dict | ui_compent.MakeOpts(["recommend_negative"], opt_type="CheckboxGroup", show_label=False)
                            ckb_grp_negative = opt_dict["recommend_negative"]
                            txt_sel_negative = gr.Textbox(label="Selected Negative Prompt", lines=5, max_lines=7)
                            opt_dict = opt_dict | ui_compent.MakeSlider(["prompt_negative_weight"])
                            opt_dict = opt_dict | ui_compent.MakeOpts(["file_format"], opt_type="Radio")
                        with gr.Column(scale=20, min_width=200):
                            ckb_step = gr.Checkbox(label="Custom Steps", elem_id="ckb_step")
                            opt_dict = opt_dict | ui_compent.MakeSlider(["step_base", "step_refiner"], interactive=False)
                            sl_step_base, sl_step_refiner = opt_dict["step_base"], opt_dict["step_refiner"]
                            ckb_size = gr.Checkbox(label="Custom Size", elem_id="ckb_size")
                            opt_dict = opt_dict | ui_compent.MakeSlider(["image_width", "image_height"], interactive=False)
                            sl_image_width, sl_image_height = opt_dict["image_width"], opt_dict["image_height"]
                            ckb_disable_style = gr.Checkbox(label="Disable Style", elem_id="ckb_disable_style")
                    opt_dict = opt_dict | {
                        "lang": opt_lang,
                        "seed": txt_seed,
                        "fixed_seed": ckb_seed,
                        "custom_step": ckb_step,
                        "custom_size": ckb_size,
                        "disable_style": ckb_disable_style,
                        "vary_model": opt_vary_model,
                        "refiner_model": opt_refiner_model,
                        "negative": txt_prompt_negative,
                    }
                # with gr.Column(visible=False):
                with gr.Tab("Setting File"):
                    with gr.Row():
                        # with gr.Column(scale=20, min_width=200):
                        txt_setting = gr.Textbox(show_label=False, lines=7, elem_id="txt_setting", visible=False)

                        with gr.Column(scale=20, min_width=200):
                            opt_dict = opt_dict | ui_compent.MakeOpts(["model_dtype", "clip_dtype", "vae_dtype", "controlnet_dtype", "ipadapter_dtype"])
                        with gr.Column(scale=20, min_width=200) as panel_setting_btns:
                            btn_download_setting = gr.Button("Download Setting", elem_id="btn_download_setting")
                            btn_load_setting = gr.UploadButton("Load Setting", file_types=[".json"])
                            btn_reset_setting = gr.Button("Reset Setting", elem_id="btn_reset_setting")

                        with gr.Column(scale=20, min_width=200):
                            img_generate_data = gr.Image(source="upload", interactive=True, type="pil", elem_id="img_generate_data")
                        
                        with gr.Column(scale=20, min_width=200):
                            txt_generate_data = gr.Textbox(label="Generate Data", lines=7, elem_id="txt_generate_data")
                            with gr.Row():
                                btn_parse_generate_data = gr.Button("Parse Data")
                                btn_generate_by_data = gr.Button("Generate")

                        img_generate_data.change(ui_process.GetImageGenerateData, img_generate_data, txt_generate_data)

            with gr.Row():
                ckb_pro = gr.Checkbox(label="Professional Mode", container=False, elem_id="ckb_pro", scale=20)
                gr.CheckboxGroup(scale=80, container=False)

                opt_dict["ckb_pro"] = ckb_pro

    opt_list = [v for k, v in opt_dict.items()]
    opt_pic_num = opt_dict['pic_num']

    ui_process.opt_dict = opt_dict
    # ui_process.ref_image_count = ref_image_count

    wooool.load(ui_process.GetSettingJson, [gl_style_list] + opt_list, txt_setting, queue=False) \
        .then(ui_process.InitSetting, txt_setting, opt_list, _js="(x => get_ui_settings(x))", queue=False) \
        .then(ui_compent.SetMCShowCount, [opt_lang, radio_mc], [radio_mc], queue=False) \
        .then(lambda:gr.Column(visible=True), None, main_ui, queue=False) \
        .then(ui_process.SetPageSize, sl_sample_pagesize, None, queue=False) \
        .then(ui_process.GetNegativeText, [ckb_grp_negative], txt_sel_negative, queue=False) \
        .then(ui_process.ChangeRefBlockNum, sl_refNum, ref_blocks, queue=False) \
        .then(None, None, [opt_lang], _js="_ => getLanguage()", queue=False)

    opt_lang.change(None, [opt_lang], None, _js="(x) => changeLanguage(x)", queue=False)
    btn_close_gallery.click(lambda: (gr.Button(visible=True), gr.Row(visible=False)), None, [title, panel_gallery], queue=False)
    gl_sample_list.select(ui_process.SelectSample, [gl_sample_list], [num_selected_sample, txt_sample_info], queue=False) \
        .then(None, txt_sample_info, None, _js="x => setModalImageInfo(x)", queue=False)
    # img_refer.change(lambda x: (gr.Image(show_label=x is None)), inputs=img_refer, outputs=[img_refer], queue=False)
    btn_prompt_lora.click(ui_process.GetLoraFromPrompt, [txt_prompt_main, opt_lora_num] + lora_list, [txt_prompt_main] + lora_list, queue=False)
    opt_dict['style'].change(lambda x: gr.Gallery(ui_process.GetStyleList(x), selected_index=0), [opt_dict['style']], gl_style_list, queue=False)
    opt_pic_num.change(None, opt_pic_num, btn_generate, _js=f"(x) => getTranslation('Generate x ' + x )", queue=False)
    sl_refNum.release(ui_process.ChangeRefBlockNum, sl_refNum, ref_blocks, queue=False)
    opt_lora_num.change(ui_process.ChangeLoraBlockNum, opt_lora_num, lora_blocks, queue=False)
    ckb_pro.change(lambda x: (gr.Row(visible=x), gr.Row(visible=not x)), ckb_pro, [panel_pro, panel_refer], queue=False)
    title.click(lambda: (gr.Button(visible=False), gr.Row(visible=True), gr.Gallery(value=ui_process.GetSampleList())), None, [title, panel_gallery, gl_sample_list], queue=False)
    btn_weight_reset.click(lambda: [gr.Slider(value=opts.default[x]) for x in opt_weights.keys()] , None, [v for _, v in opt_weights.items()], queue=False)
    opt_base_model.change(ui_process.GetModelInfo, [opt_base_model], [btn_base_model_info], _js="x => model_link(x)", queue=False)
    ckb_seed.change(lambda x: (gr.Textbox(interactive=x), gr.Slider(visible=x)), ckb_seed, [txt_seed, sl_subseed_strength], queue=False)
    ckb_step.change(lambda x: (gr.Slider(interactive=x), gr.Slider(interactive=x)), ckb_step, [sl_step_base, sl_step_refiner], queue=False)
    ckb_size.change(lambda x: (gr.Slider(interactive=x), gr.Slider(interactive=x), gr.Dropdown(interactive=not x)), ckb_size, [sl_image_width, sl_image_height, opt_ratios], queue=False)
    ckb_grp_negative.change(ui_process.GetNegativeText, ckb_grp_negative, txt_sel_negative, queue=False)
    sl_sample_pagesize.release(ui_process.ChangePageSize, sl_sample_pagesize, [gl_sample_list, num_page_sample], queue=False)
    ckb_disable_style.change(lambda x: (gr.Dropdown(interactive=not x), gr.Gallery(visible=not x)), ckb_disable_style, [opt_style, gl_style_list], queue=False)
    ckb_endless_mode.change(ui_process.ChangeEndless, ckb_endless_mode, None, queue=False)
    btn_refresh_lora.click(ui_process.RefreshLoras, None, lora_list[::4], queue=False)

    for x in opt_list:
        opt_type = str(x)
        if opt_type in ["slider"]:
            opt_evt = x.release
        elif opt_type in ["textbox", "textarea"]:
            opt_evt = x.input
        else:
            opt_evt = x.change

        opt_evt(ui_process.GetSettingJson, [gl_style_list] + opt_list, txt_setting, queue=False) \
            .then(None, txt_setting, None, _js="(x) => save_ui_settings(x)", queue=False)

    generate_inputs = [img_refer, ckb_pro, txt_setting] + ref_list + lora_list
    action_inputs = [gl_sample_list, num_selected_sample, txt_setting]
    action_btns = [btn_vary_subtle, btn_vary_strong, btn_vary_custom, btn_vary_custom_interface, btn_zoom_out15, btn_zoom_out20, btn_zoom_custom_interface, btn_change_style, btn_resize_interface, btn_resize, btn_reface_interface, btn_refiner, btn_refiner_interface, btn_refiner_face, btn_refiner_face_interface, btn_upscale, btn_upscale_interface, btn_delete]
    generate_outpus_base = [title, html_proccbar, panel_gallery, panel_generate, panel_process, btn_skip, btn_stop]
    generate_outpus = generate_outpus_base + [gl_sample_list] + action_btns
    ui_process.action_btns = action_btns

    btn_generate.click(fn=ui_process.ChangeSeed, inputs=ckb_seed, outputs=txt_seed, queue=False) \
        .then(fn=ui_process.GetSettingJson, inputs=[gl_style_list] + opt_list, outputs=txt_setting, queue=False) \
        .then(lambda: gr.Checkbox(visible=True), None, ckb_endless_mode, queue=False) \
        .then(fn=ui_process.Generate, inputs=generate_inputs, outputs=generate_outpus) \
        .then(lambda x: gr.Checkbox(visible=x), ckb_endless_mode, ckb_endless_mode, _js="(x) => endless(x)", queue=False) \
        .then(ui_process.ProcessFinishNoRefresh, inputs=ckb_endless_mode, outputs=generate_outpus_base, queue=False)
    
    btn_generate_one.click(fn=ui_process.ChangeSeed, inputs=ckb_seed, outputs=txt_seed, queue=False) \
        .then(fn=ui_process.GetSettingJson, inputs=[gl_style_list] + opt_list, outputs=txt_setting, queue=False) \
        .then(fn=ui_process.GenerateOne, inputs=generate_inputs, outputs=generate_outpus)
    
    btn_skip.click(fn=ui_process.SkipBatch, inputs=None, outputs=[btn_skip], queue=False)
    btn_stop.click(fn=ui_process.StopProcess, inputs=None, outputs=[btn_skip, btn_stop, ckb_endless_mode], queue=False)

    btn_vary_subtle.click(ui_process.Process(ui_action.VarySubtle), action_inputs, generate_outpus)
    btn_vary_strong.click(ui_process.Process(ui_action.VaryStrong), action_inputs, generate_outpus)
    btn_vary_custom.click(ui_process.Process(ui_action.VaryCustom), action_inputs + [img_vary_editor], generate_outpus)
    btn_zoom_out15.click(ui_process.Process(ui_action.ZoomOut), action_inputs, generate_outpus)
    btn_zoom_out20.click(ui_process.Process(ui_action.ZoomOut20), action_inputs, generate_outpus)
    btn_zoom_custom.click(ui_process.Process(ui_action.ZoomOutCustom), action_inputs, generate_outpus)
    btn_change_style.click(ui_process.Process(ui_action.ChangeStyle), action_inputs + [gl_style_list, num_selected_style], generate_outpus)
    btn_resize.click(ui_process.Process(ui_action.Resize), action_inputs, generate_outpus)
    btn_reface.click(ui_process.Process(ui_action.ReFace), action_inputs + [img_face], generate_outpus)
    btn_refiner.click(ui_process.Process(ui_action.Refiner), action_inputs, generate_outpus)
    btn_refiner_face.click(ui_process.Process(ui_action.RefinerFace), action_inputs, generate_outpus)
    btn_upscale.click(ui_process.Process(ui_action.Upscale), action_inputs, generate_outpus)

    btn_delete.click(ui_process.DeleteSample, [gl_sample_list, num_selected_sample, num_page_sample], [gl_sample_list], _js="(x, y, z) => deleteSample(x, y, z)", queue=False)
    btn_top.click(ui_process.TopSample, [gl_sample_list, num_selected_sample, num_page_sample], [gl_sample_list], queue=False)

    btn_first.click(ui_process.FirstPageSample, [num_page_sample], [gl_sample_list, num_page_sample], queue=False)
    btn_prev.click(ui_process.PrevPageSample, [num_page_sample], [gl_sample_list, num_page_sample], queue=False)
    btn_next.click(ui_process.NextPageSample, [num_page_sample], [gl_sample_list, num_page_sample], queue=False)

    btn_download_setting.click(None, txt_setting, None, _js="(x => download_ui_settings(x))", queue=False)
    btn_load_setting.upload(ui_process.LoadSetting, btn_load_setting, txt_setting, queue=False) \
        .then(ui_process.InitSetting, txt_setting, opt_list, queue=False)
    btn_reset_setting.click(ui_process.ResetSetting, None, opt_list, _js="(x => reset_ui_settings(x))", queue=False)

    btn_parse_generate_data.click(ui_process.ParseGenerateData, txt_generate_data, txt_setting, queue=False) \
        .then(ui_process.InitSetting, txt_setting, opt_list, queue=False)
    btn_generate_by_data.click(ui_process.GenerateByData, [txt_generate_data, txt_setting], generate_outpus)

wooool.launch(
    server_name=shared.args.listen,
    server_port=shared.args.port,
    inbrowser=shared.args.auto_launch,
)