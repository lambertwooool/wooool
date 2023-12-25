
// localization = {} -- the dict with translations is created by the backend

var ignore_ids_for_localization = {
   
};

var re_num = /^[.\d]+$/;
var re_emoji = /[\p{Extended_Pictographic}\u{1F3FB}-\u{1F3FF}\u{1F9B0}-\u{1F9B3}]/u;

var original_lines = {};
var translated_lines = {};

async function hasLocalization() {
    return (window.localization || await fetchlocalization()) && Object.keys(window.localization).length > 0;
}

async function fetchlocalization(lang) {
    lang = lang || localStorage["lang"] || navigator.language;
    let localization;
    url = (window.localization_url[lang] || window.localization_url['en']).url;
    if (url) {
        response = await fetch(url);
        localization_data = await response.json();
        localization = {}
        for (var k in window.localization_url) {
            localization[k.toLowerCase()] = window.localization_url[k].name;
        }
        for (var k in localization_data) {
            localization[k.toLowerCase()] = localization_data[k]
        }
        window.localization = localization;
        localStorage["lang"] = lang;
    }
    return localization;
}

function getLanguage() {
    return localStorage["lang"] || "en";
}

function changeLanguage(lang) {
    if (lang && window.localization_url[lang] && localStorage["lang"] != lang) {
        localStorage["lang"] = lang;
        window.location.reload(true);
    }
}

function textNodesUnder(el) {
    var n, a = [], walk = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null, false);
    while ((n = walk.nextNode())) a.push(n);
    return a;
}

function canBeTranslated(node, text) {
    if (!text) return false;
    if (!node.parentElement) return false;

    var parentType = node.parentElement.nodeName;
    if (parentType == 'SCRIPT' || parentType == 'STYLE' || parentType == 'TEXTAREA') return false;

    if (parentType == 'OPTION' || parentType == 'SPAN') {
        var pnode = node;
        for (var level = 0; level < 4; level++) {
            pnode = pnode.parentElement;
            if (!pnode) break;

            if (ignore_ids_for_localization[pnode.id] == parentType) return false;
        }
    }

    // if (re_num.test(text)) return false;
    // if (re_emoji.test(text)) return false;
    return true;
}

function getTranslation(text) {
    if (!text) return undefined;
    text = text.toLowerCase()

    if (translated_lines[text] === undefined) {
        original_lines[text] = 1;
    }

    var tl = localization[text];
    if (tl !== undefined) {
        translated_lines[tl] = 1;
    }

    return tl;
}

function processTextNode(node) {
    var text = node.textContent.trim();

    if (!canBeTranslated(node, text)) return;

    tl = getTranslation(text)
    if (tl === undefined){
        tl = text;
        text.split(/[^a-z ]+/i).forEach((t) => {
            var tll = getTranslation(t.trim());
            if (tll !== undefined) {
                tl = tl.replace(t, tll);
            }
        });
    }
    node.textContent = tl;
}

function processNode(node) {
    if (node.nodeType == 3) {
        processTextNode(node);
        return;
    }

    if (node.title) {
        let tl = getTranslation(node.title);
        if (tl !== undefined) {
            node.title = tl;
        }
    }

    if (node.placeholder) {
        let tl = getTranslation(node.placeholder);
        if (tl !== undefined) {
            node.placeholder = tl;
        }
    }

    textNodesUnder(node).forEach(function(node) {
        processTextNode(node);
    });
}

function localizeWholePage() {
    function elem(comp) {
        var elem_id = comp.props.elem_id ? comp.props.elem_id : "component-" + comp.id;
        return gradioApp().getElementById(elem_id);
    }

    for (var comp of window.gradio_config.components) {
        if (comp.props.elem_id == "opt_lang") {
            comp.props.value = localStorage["lang"] || "en";
        }

        if (comp.props.webui_tooltip) {
            let e = elem(comp);

            let tl = e ? getTranslation(e.title) : undefined;
            if (tl !== undefined) {
                e.title = tl;
            }
        }
        if (comp.props.placeholder) {
            let e = elem(comp);
            let textbox = e ? e.querySelector('[placeholder]') : null;

            let tl = textbox ? getTranslation(textbox.placeholder) : undefined;
            if (tl !== undefined) {
                textbox.placeholder = tl;
            }
        }
    }
}

document.addEventListener("DOMContentLoaded", async function() {
    if (!(await hasLocalization())) {
        return;
    }

    function updateValue(el) {
        dp_input = el.srcElement
        dp_input.style.visibility = "hidden";
        setTimeout(() => {
            tl = getTranslation(dp_input.value);
            if (tl) {
                dp_input.value = tl;
            }
            dp_input.style.visibility = "";
        }, 1)
    }

    onUiUpdate(function(m) {
        dp_inputs = document.querySelectorAll(".gr_dropdown input", gradioApp());
        dp_inputs.forEach(function(dp_input) {
            dp_input.addEventListener("blur", updateValue);

            tl = getTranslation(dp_input.value);
            if (tl) {
                dp_input.value = tl;
            }
        });

        m.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                processNode(node);
            });
        });
        processNode(gradioApp());
    });

    processNode(gradioApp());
    localizeWholePage();

    if (localization.rtl) { // if the language is from right to left,
        (new MutationObserver((mutations, observer) => { // wait for the style to load
            mutations.forEach(mutation => {
                mutation.addedNodes.forEach(node => {
                    if (node.tagName === 'STYLE') {
                        observer.disconnect();

                        for (const x of node.sheet.rules) { // find all rtl media rules
                            if (Array.from(x.media || []).includes('rtl')) {
                                x.media.appendMedium('all'); // enable them
                            }
                        }
                    }
                });
            });
        })).observe(gradioApp(), {childList: true});
    }
});
