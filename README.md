# Wooool web UI

Wooool is an AIGC image generator, Refer to the [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 、Midjourney and [Fooocus](https://github.com/lllyasviel/Fooocus) part of the function, support SD1.5 and SDXL model

![](screenshot.png)

## Features

### Prompt
More use of choice, rather than filling in the blank, the use of photographic description, gradually form a picture description
- Main Character: provides more than 40 types of protagonists by default. Each refresh displays a group of protagonists. For Girl and Male, different races are displayed according to the selected language, so as to conform to the aesthetics of different regions
- Integrated [Dynamic Prompts](https://github.com/adieyal/dynamicprompts), supporting rich prompt word functions such as random selection
- Supports prompt word weights
- You can choose content such as perspective, emo, location, weather and lighting, which is more suitable for photography
- SDXL Style selection, with display images configured for each style
- Weights can be set individually for each prompt content such as Main Character and SDXL style

### Reference Image
Anything is a reference image
- IPAdapter，Reference All(IPAdapter) 和 Face only(IPAdapter Face)
- Controlnet，Layout only(Canny)、Depth only(Depth)、Pose only(DWPose)
- Wd14Tagger, Content only
- Base Image, Img2img

### Fine tune (Lora Model)
Integrate the Civitai API to quickly preview and switch trigger words

### Actions
- Vary(Subtle) and Vary(Strong), Midjourney Vary.
- Zoom(1.5x) and Zoom(2.0x), Midjourney Zoom
- Change Style, Transfer styles between SDXL styles
- Resize, switch between different sizes, similar to Photoshop fill
- Refiner(Image)，Faces are protected. The proportion of faces is too small
- Refiner(Face)，similar to the After Detail plug-in, the face is enlarged and refined
- ReFace，replace face
- Upscale，4x in HD

### Gallery
Integrated image browser and generate preview for easier management of generated images

### Performance
- The multi-threading mode enables the CPU and GPU to share the processing power and process prompt words, picture generation and VAE in parallel, thus reducing video memory switching and speeding up the overall speed

### Configuate
- config/user_paths.json, configure the model and the file directory
- config/download.json, configures the download address of the model
- config/localization, multi-language configuration
- config/sd_styles, Indicates the configuration of SDXL

### Installation

```bash
python launch.py
```

### License and Acknowledgement

This project is released under the MIT license.

### :e-mail: Contact

If you have any question, open an issue or email `kevinwangling@gmail.com`.