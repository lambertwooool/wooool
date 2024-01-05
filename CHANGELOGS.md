# Change Logs

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