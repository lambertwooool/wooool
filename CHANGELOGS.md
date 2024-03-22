# Change Logs

## 0.8.8
- Support sdxl and sd15 lightning generate, 6 step generate an image
- Support sdxl and sd15 tcd generate and eta weight, 8 step generate an image
- Support playground V2.5 generate
- Support layered diffusion
- Add sdxl tile controlnet
- Optimize the CFG algorithm and add CFG scaling
- Apply freeU optimize image
- Apply differential diffusion optimize vary, zoom and resize

## 0.8.7
- Base and Controlnet Model use FP8
- Support instant id

## 0.8.6
- Support new controlnet, depth anythiny, teed
- controlnet annotator cache
- Support new IPAdapters faceid and portrait
- IPAdapter can use the start and end percent
- Better integration of multiple IPAdapters
- Use style aligned when generating multiple images

## 0.8.5
- Inpaint adds vary after erase
- Controlnet supports multiple preprocessors
- Vary use inpaint controlnet at sd15 and Refiner use tile controlnet at sd15
- Any combination of sd15 and sdxl, used as a base model or refiner model

## 0.8.4
- New finetune interface, supports extracting Lora from prompt words into finetune list, as well as selecting number of Lora models
- Enable or disable individual ref image or Lora model
- Upscale UI, including upscale factor, model selection, face restoration etc. options
- Image zoom interface after double click
- Vary(Custom) Inpaint UI
- New endless mode, can adjust parameters in real time during generation

## 0.8.3
- Support more controlnet, densepose, animal_pose, lineart, lineart_anime, manga_line, mlsd, normalbae, oneformer, sam, tile, depth_zoe, qrcode, brightness, recolor, color, style.
- Controlnet can adjust the starting and ending percent, as well as preprocessor selection.
- Supports reading pnginfo and can directly generate images or parse them into interface options.
- Added some styles and protagonists.
- Supports TorchScript models.
- Refiner and Refiner face interface (initial version).
- Updated pnginfo format.
- In the prompt, you can use {__mc__} to specify the location of the protagonist's prompt word appearance.

## 0.8.2
- Add 8 Wooool exclusive styles, Marble sculpture, Matrix, Underwater World, etc
- Add new protagonists and locations
- Optimize VRAM processing
- Add the SDXL Turbo model configuration in configs/settings/sdxl-turbo.json
- Fine-tune supports artistic control, use Lora xl_more_art-full_v1
- Support zoom(custom), rewrite the extended base image algorithm, and the effect is significantly improved
- Resize and Refine(face) use new UI

## 0.8.1
- Checkpoints and Loras can be directly linked to the model page on civitai
- Lora models can be added in the profiles of protagonist, style, location, etc.
- Supports random style selection
- Supports Vary(Custom), i.e. inpaint function
- Configuration options can be saved as templates and loaded anytime
- Different Controlnet models can be selected under control operations
- Detail enhancement slider
- Fixes the default value issue of dropdown menus
- Supports custom steps and sizes

## 0.8.0
first commit