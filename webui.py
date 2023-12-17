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
    opt_lang = gr.Dropdown(choices=localization.urls().keys(), value="en", container=False, filterable=False, min_width=50, elem_id=f"opt_lang", elem_classes="gr_drapdown")
    with gr.Column(visible=False) as main_ui:
        with gr.Column(elem_id="panel_header") as panel_header:
            title = gr.Button(f"{app_name}", visible=True, elem_id="title")
            
            with gr.Row(elem_id="panel_gallery", visible=False) as panel_gallery:
                gl_sample_list, num_selected_sample, txt_sample_info = ui_compent.SampleGallery()

                with gr.Column(elem_id="panel_action", min_width=260) as panel_action:
                    with gr.Row():
                        ckb_fullscreen = gr.Checkbox(label="Full Screen", min_width=100, scale=3)
                        btn_close_gallery = gr.Button("Close", size="sm", min_width=80, elem_id="btn_close_gallery", scale=1)
                    with gr.Tab("Action"):
                        with gr.Column() as panel_action_btns:
                            with gr.Row():
                                btn_vary_subtle = gr.Button("Vary(Subtle)", min_width=100, elem_id="btn_vary_subtle", elem_classes="btn_action")
                                btn_vary_strong = gr.Button("Vary(Strong)", min_width=100, elem_id="btn_vary_strong", elem_classes="btn_action")
                                btn_zoom_out15 = gr.Button("Zoom(1.5x)", min_width=100, elem_id="btn_zoom_out15", elem_classes="btn_action")
                                btn_zoom_out20 = gr.Button("Zoom(2.0x)", min_width=100, elem_id="btn_zoom_out20", elem_classes="btn_action")
                                btn_change_style = gr.Button("Change Style", min_width=100, elem_id="btn_change_style", elem_classes="btn_action")
                                btn_resize_interface = gr.Button("ReSize", min_width=100, elem_id="btn_resize", elem_classes="btn_action")
                                btn_refiner = gr.Button("Refiner Image", min_width=100, elem_id="btn_refiner", elem_classes="btn_action")
                                btn_refiner_face = gr.Button("Refiner Face", min_width=100, elem_id="btn_refiner_face", elem_classes="btn_action")
                                btn_reface_interface = gr.Button("ReFace", min_width=100, elem_id="btn_reface_interface", elem_classes="btn_action")

                                btn_upscale = gr.Button("Upscale", min_width=100, elem_classes="btn_action")
                                # gr.Button("Change BG", min_width=100, elem_classes="btn_action"
                            
                            with gr.Row():
                                btn_delete = gr.Button("Delete", min_width=100, size="sm", elem_id="btn_delete", elem_classes="btn_action")
                                btn_top = gr.Button("Set to top", min_width=100, size="sm", elem_id="btn_top", elem_classes="btn_action")

                            with gr.Row():
                                btn_first = gr.Button("First page", min_width=60, size="sm", elem_id="btn_prev", elem_classes="btn_action")
                                btn_prev = gr.Button("Prev page", min_width=60, size="sm", elem_id="btn_prev", elem_classes="btn_action")
                                btn_next = gr.Button("Next page", min_width=60, size="sm", elem_id="btn_next", elem_classes="btn_action")
                                num_page_sample = gr.Number(step=1, minimum=0, min_width=100, visible=False, elem_id="btn_next")
                        
                        with gr.Row(visible=False) as panel_reface:
                            img_face = gr.Image(label="ReFace", show_label=False, width=200, height=160, min_width=200, elem_id="img_face")
                            btn_reface = gr.Button("ReFace", min_width=100, variant="primary", interactive=False, elem_id="btn_reface", elem_classes="btn_action")
                            btn_reface_cancel = gr.Button("Close", min_width=100, elem_id="btn_reface_cancel", elem_classes="btn_action")

                        img_face.change(lambda x: gr.Button(interactive=x is not None), img_face, btn_reface)
                        btn_reface_interface.click(lambda:(gr.Column(visible=False), gr.Row(visible=True)), None, [panel_action_btns, panel_reface], queue=False)
                        btn_reface_cancel.click(lambda:(gr.Column(visible=True), gr.Row(visible=False)), None, [panel_action_btns, panel_reface], queue=False)

                        with gr.Row(visible=False) as panel_resize:
                            opt_dict["resize_ratios"] = ui_compent.MakeOpts(["ratios"], min_width=160)["ratios"]
                            opt_resize = opt_dict["resize_ratios"]
                            btn_resize = gr.Button("ReSize", min_width=100, variant="primary", elem_id="btn_resize", elem_classes="btn_action")
                            btn_resize_cancel = gr.Button("Close", min_width=100, elem_id="btn_reface_cancel", elem_classes="btn_action")
                        
                        btn_resize_interface.click(lambda:(gr.Column(visible=False), gr.Row(visible=True)), None, [panel_action_btns, panel_resize], queue=False)
                        btn_resize_cancel.click(lambda:(gr.Column(visible=True), gr.Row(visible=False)), None, [panel_action_btns, panel_resize], queue=False)

                    with gr.Tab("Setting"):
                        opt_vary_model = gr.Dropdown(choices=ui_process.GetRefinerModels("sdxl"), value=model_helper.default_base_name, label="Vary & Zoom Model", filterable=False, elem_id=f"opt_vary_model", elem_classes="gr_drapdown")
                        opt_dict = opt_dict | ui_compent.MakeSlider(["sample_pagesize"])
                        sl_sample_pagesize = opt_dict["sample_pagesize"]
        
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
                # btn_more_generate = gr.Button("Generate", visible=True, size="sm", min_width=80, elem_id="btn_more_generate", scale=5)
                btn_skip = gr.Button("Skip", visible=True, variant="stop", size="sm", min_width=80, elem_id="btn_skip", scale=5)
                btn_stop = gr.Button("Stop", visible=True, variant="stop", size="sm", min_width=80, elem_id="btn_stop", scale=5)

            with gr.Row() as panel_generate:
                opt_dict = opt_dict | ui_compent.MakeOpts(['pic_num'], show_label=False, container=False, scale=10)
                btn_generate = gr.Button(f"Generate x {opts.default['pic_num']}", variant="primary", elem_id="btn_generate", scale=80)

            with gr.Column():
                with gr.Row():
                    opt_dict = opt_dict | ui_compent.MakeOpts(['style', 'ratios', 'view', 'emo', 'location', 'weather', 'hue'])
                
                gl_style_list, num_selected_style = ui_compent.StyleGallery()
                opt_dict = opt_dict | {
                    "style_index": num_selected_style
                }

            with gr.Column(visible=False) as panel_pro:
                with gr.Tab("General"):
                    with gr.Row(elem_id="panel_pro_general") as panel_pro_general:
                        with gr.Column(scale=20, min_width=200):
                            base_models = list(model_helper.base_files)
                            opt_base_model = gr.Dropdown(choices=base_models, value=model_helper.default_base_name, label="Draw Model", filterable=False, elem_id=f"opt_base_model", elem_classes="gr_drapdown")
                            opt_dict = opt_dict | {
                                "base_model": opt_base_model,
                            } | ui_compent.MakeSlider(['ref_num']) | ui_compent.MakeOpts(["quality"])
                            sl_refNum = opt_dict['ref_num']

                        with gr.Column(scale=80):
                            with gr.Row():
                                ref_blocks, ref_list = ui_compent.RefBlock(*ui_process.ref_image_count)
                            with gr.Row():
                                gr.CheckboxGroup(container=False)
                                gr.CheckboxGroup(container=False)
                with gr.Tab("Fine tune"):
                    with gr.Row():
                        lora_blocks, lora_list = ui_compent.LoraBlock(*ui_process.lora_count)
                with gr.Tab("Advanced"):
                    with gr.Row(elem_id="panel_pro_advanced") as panel_pro_advanced:
                        with gr.Column(scale=20, min_width=200):
                            opt_dict = opt_dict | ui_compent.MakeOpts(["simpler", "scheduler", "detail"])
                            opt_dict = opt_dict | ui_compent.MakeSlider(["cfg", "cfg_to", "clip_skip"])
                            opt_dict = opt_dict | ui_compent.MakeOpts(["file_format"], opt_type="Radio")
                        with gr.Column(scale=20, min_width=200):
                            with gr.Row():
                                txt_seed = gr.Number(label="Seed", value=0, elem_id="txt_seed", interactive=False, min_width=80, scale=80)
                                ckb_seed = gr.Checkbox(label="Fixed", value=False, elem_id="ckb_seed", min_width=80, scale=20)
                            opt_dict = opt_dict | ui_compent.MakeSlider(["subseed_strength"], visible=False)
                            sl_subseed_strength = opt_dict["subseed_strength"]
                            
                            opt_refiner_model = gr.Dropdown(choices=[opts.title["disable_refiner"]] + ui_process.GetRefinerModels("sdxl"), value=model_helper.default_refiner_name or opts.title["disable_refiner"], label="Refiner Model", filterable=False, elem_id=f"opt_refiner_model", elem_classes="gr_drapdown")
                            btn_refresh_model = gr.Button("🔄 Refresh Models", elem_id="btn_refresh_model")
                            opt_dict = opt_dict | ui_compent.MakeSlider(["denoise"])
                            btn_refresh_model.click(ui_process.RefreshModels, None, [opt_base_model, opt_vary_model, opt_refiner_model])
                            opt_dict = opt_dict | ui_compent.MakeSlider(["step_scale", "refiner_step_scale"])
                            opt_dict = opt_dict | ui_compent.MakeOpts(["single_vae"], opt_type="Radio")
                        with gr.Column(scale=20, min_width=200):
                            txt_prompt_negative = gr.Textbox(label="Negative Prompt", lines=3, placeholder="Input negative.", elem_id="txt_prompt_negative", elem_classes="prompt")
                            opt_dict = opt_dict | ui_compent.MakeOpts(["recommend_negative"], opt_type="CheckboxGroup", show_label=False)
                            ckb_grp_negative = opt_dict["recommend_negative"]
                            txt_sel_negative = gr.Textbox(label="Selected Negative Prompt", lines=5, max_lines=7)
                            opt_dict = opt_dict | ui_compent.MakeSlider(["prompt_negative_weight"])
                        with gr.Column(scale=20, min_width=200):
                            opt_weights = ui_compent.MakeSlider(["mc_weight", "style_weight", "view_weight", "emo_weight", "location_weight", "weather_weight", "hue_weight"])
                            opt_dict = opt_dict | opt_weights
                            btn_weight_reset = gr.Button("Weight Reset", elem_id="btn_weight_reset")
                    opt_dict = opt_dict | {
                        "lang": opt_lang,
                        "seed": txt_seed,
                        "fixed_seed": ckb_seed,
                        "vary_model": opt_vary_model,
                        "refiner_model": opt_refiner_model,
                        "negative": txt_prompt_negative,
                    }
                with gr.Column(visible=False):
                # with gr.Tab("Setting JSON"):
                    txt_setting = gr.Textbox(show_label=False, lines=7, elem_id="txt_setting")

            with gr.Row():
                ckb_pro = gr.Checkbox(label="Professional Mode", container=False, elem_id="ckb_pro", scale=20)
                gr.CheckboxGroup(scale=80, container=False)

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
    btn_close_gallery.click(lambda: (gr.Button(visible=True), gr.Row(visible=False), gr.Checkbox(value=False)), None, [title, panel_gallery, ckb_fullscreen], queue=False)
    gl_sample_list.select(ui_process.SelectSample, [gl_sample_list, ckb_fullscreen], [gl_sample_list, num_selected_sample, txt_sample_info], queue=False)
    ckb_fullscreen.change(lambda x: (gr.Column(visible=not x), gr.Gallery(height=866 if x else 444), gr.Text(visible=x)), ckb_fullscreen, [panel_main, gl_sample_list, txt_sample_info], queue=False)
    # img_refer.change(lambda x: (gr.Image(show_label=x is None)), inputs=img_refer, outputs=[img_refer], queue=False)
    opt_dict['style'].change(lambda x: gr.Gallery(ui_process.GetStyleList(x), selected_index=0), [opt_dict['style']], gl_style_list, queue=False)
    opt_pic_num.change(None, opt_pic_num, btn_generate, _js=f"(x) => getTranslation('Generate x ' + x )", queue=False)
    sl_refNum.release(ui_process.ChangeRefBlockNum, sl_refNum, ref_blocks, queue=False)
    ckb_pro.change(lambda x: (gr.Row(visible=x), gr.Row(visible=not x)), ckb_pro, [panel_pro, panel_refer], queue=False)
    title.click(lambda: (gr.Button(visible=False), gr.Row(visible=True), gr.Gallery(value=ui_process.GetSampleList())), None, [title, panel_gallery, gl_sample_list], queue=False)
    btn_weight_reset.click(lambda: [gr.Slider(value=100)] * len(opt_weights.keys()) , None, [v for _, v in opt_weights.items()], queue=False)
    # opt_base_model.change(lambda x: gr.Dropdown(choices=ui_process.GetRefinerModels(x)), [opt_base_model], [opt_refiner_model])
    ckb_seed.change(lambda x: (gr.Textbox(interactive=x), gr.Slider(visible=x)), ckb_seed, [txt_seed, sl_subseed_strength], queue=False)
    ckb_grp_negative.change(ui_process.GetNegativeText, ckb_grp_negative, txt_sel_negative, queue=False)
    sl_sample_pagesize.release(ui_process.ChangePageSize, sl_sample_pagesize, [gl_sample_list, num_page_sample], queue=False)

    for x in opt_list:
        x.change(ui_process.GetSettingJson, [gl_style_list] + opt_list, txt_setting, queue=False) \
            .then(None, txt_setting, None, _js="(x) => save_ui_settings(x)", queue=False)

    generate_inputs = [img_refer, ckb_pro, txt_setting] + ref_list + lora_list
    action_inputs = [gl_sample_list, num_selected_sample, txt_setting]
    action_btns = [btn_vary_subtle, btn_vary_strong, btn_zoom_out15, btn_zoom_out20, btn_change_style, btn_resize_interface, btn_resize, btn_reface_interface, btn_refiner, btn_refiner_face, btn_upscale]
    generate_outpus = [title, html_proccbar, gl_sample_list, panel_gallery, panel_generate, panel_process, btn_skip, btn_stop, btn_delete] + action_btns
    ui_process.action_btns = action_btns

    btn_generate.click(fn=ui_process.ChangeSeed, inputs=ckb_seed, outputs=txt_seed, queue=False) \
        .then(fn=ui_process.GetSettingJson, inputs=[gl_style_list] + opt_list, outputs=txt_setting, queue=False) \
        .then(fn=ui_process.Generate, inputs=generate_inputs, outputs=generate_outpus)
    
    # btn_more_generate.click(fn=ui_process.ChangeSeed, inputs=ckb_seed, outputs=txt_seed, queue=False) \
    #     .then(fn=ui_process.Generate, inputs=generate_inputs, outputs=generate_outpus)
    
    btn_skip.click(fn=ui_process.SkipBatch, inputs=None, outputs=[btn_skip], queue=False)
    btn_stop.click(fn=ui_process.StopProcess, inputs=None, outputs=[btn_skip, btn_stop], queue=False)

    btn_vary_subtle.click(ui_process.Process(ui_action.VarySubtle), action_inputs, generate_outpus)
    btn_vary_strong.click(ui_process.Process(ui_action.VaryStrong), action_inputs, generate_outpus)
    btn_zoom_out15.click(ui_process.Process(ui_action.ZoomOut), action_inputs, generate_outpus)
    btn_zoom_out20.click(ui_process.Process(ui_action.ZoomOut20), action_inputs, generate_outpus)
    btn_change_style.click(ui_process.Process(ui_action.ChangeStyle), action_inputs + [gl_style_list, num_selected_style], generate_outpus)
    btn_resize.click(ui_process.Process(ui_action.Resize), action_inputs, generate_outpus)
    btn_reface.click(ui_process.Process(ui_action.ReFace), action_inputs + [img_face], generate_outpus)
    btn_refiner.click(ui_process.Process(ui_action.Refiner), action_inputs, generate_outpus)
    btn_refiner_face.click(ui_process.Process(ui_action.RefinerFace), action_inputs, generate_outpus)
    btn_upscale.click(ui_process.Process(ui_action.Upscale), action_inputs, generate_outpus)

    btn_delete.click(ui_process.DeleteSample, [gl_sample_list, num_selected_sample, num_page_sample], [gl_sample_list], queue=False)
    btn_top.click(ui_process.TopSample, [gl_sample_list, num_selected_sample, num_page_sample], [gl_sample_list], queue=False)

    btn_first.click(ui_process.FirstPageSample, [num_page_sample], [gl_sample_list, num_page_sample], queue=False)
    btn_prev.click(ui_process.PrevPageSample, [num_page_sample], [gl_sample_list, num_page_sample], queue=False)
    btn_next.click(ui_process.NextPageSample, [num_page_sample], [gl_sample_list, num_page_sample], queue=False)


wooool.launch(
    server_name=shared.args.listen,
    server_port=shared.args.port,
    inbrowser=shared.args.auto_launch,
)