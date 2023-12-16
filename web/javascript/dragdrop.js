// allows drag-dropping files into gradio image elements, and also pasting images from clipboard

function isValidImageList(files) {
    return files && files?.length === 1 && ['image/png', 'image/gif', 'image/jpeg'].includes(files[0].type);
}

function dropReplaceImage(imgWrap, files) {
    if (!isValidImageList(files)) {
        return;
    }

    const tmpFile = files[0];

    imgWrap.querySelector('.modify-upload button + button, .touch-none + div button + button')?.click();
    const callback = () => {
        const fileInput = imgWrap.querySelector('input[type="file"]');
        if (fileInput) {
            if (files.length === 0) {
                files = new DataTransfer();
                files.items.add(tmpFile);
                fileInput.files = files.files;
            } else {
                fileInput.files = files;
            }
            fileInput.dispatchEvent(new Event('change'));
        }
    };

    window.requestAnimationFrame(() => callback());
}

function eventHasFiles(e) {
    if (!e.dataTransfer || !e.dataTransfer.files) return false;
    if (e.dataTransfer.files.length > 0) return true;
    if (e.dataTransfer.items.length > 0 && e.dataTransfer.items[0].kind == "file") return true;

    return false;
}

function dragDropTargetIsPrompt(target) {
    // if (target?.placeholder && target?.placeholder.indexOf("Prompt") >= 0) return true;
    // if (target?.parentNode?.parentNode?.className?.indexOf("prompt") > 0) return true;
    // return false;
    return true;
}

window.document.addEventListener('dragover', e => {
    const target = e.composedPath()[0];
    if (!eventHasFiles(e)) return;

    var targetImage = target.closest('[data-testid="image"]');
    if (!dragDropTargetIsPrompt(target) && !targetImage) return;

    e.stopPropagation();
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
});

window.document.addEventListener('drop', e => {
    const target = e.composedPath()[0];
    if (!eventHasFiles(e)) return;
    if (dragDropTargetIsPrompt(target)) {
        e.stopPropagation();
        e.preventDefault();

        // let prompt_target = get_tab_index('tabs') == 1 ? "img2img_prompt_image" : "txt2img_prompt_image";

        // const imgParent = gradioApp().getElementById(prompt_target);
        imgParent = gradioApp()
        const files = e.dataTransfer.files;
        const fileInput = imgParent.querySelector('input[type="file"]');

        if (fileInput) {
            fileInput.files = files;
            fileInput.dispatchEvent(new Event('change'));
        }
    }

    var targetImage = target.closest('[data-testid="image"]');
    if (targetImage) {
        e.stopPropagation();
        e.preventDefault();
        const files = e.dataTransfer.files;
        dropReplaceImage(targetImage, files);
        return;
    }
});

window.addEventListener('paste', e => {
    const files = e.clipboardData.files;
    if (!isValidImageList(files)) {
        return;
    }

    const visibleImageFields = [...gradioApp().querySelectorAll('[data-testid="image"]')]
        .filter(el => uiElementIsVisible(el))
        .sort((a, b) => uiElementInSight(b) - uiElementInSight(a));


    if (!visibleImageFields.length) {
        return;
    }

    let firstFreeImageField = visibleImageFields
        .filter(el => el.querySelector('input[type=file]') &&
            ((document.activeElement && el.parentNode.contains(document.activeElement)) ||
            !el.querySelector('img[loading="lazy"]')))?.[0];

    dropReplaceImage(
        firstFreeImageField ?
            firstFreeImageField :
            visibleImageFields[visibleImageFields.length - 1]
        , files
    );
});
