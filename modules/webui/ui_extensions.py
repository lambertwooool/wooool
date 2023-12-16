import json
import os
import gradio as gr
import modules.paths
from modules import shared, localization
from modules.util import list_files, webpath

GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse

def javascript_html():
    head = ""

    # localization_path = f"{modules.paths.localization_path}/{shared.localization}.json";
    head += f'<script type="text/javascript">window.localization_url = {json.dumps(localization.urls())}</script>'

    for script in list_files(modules.paths.scripts_path, "js"):
        head += f'<script type="text/javascript" src="{webpath(script)}"></script>\n'

    return head

def css_html():
    head = ""

    for css in list_files(modules.paths.css_path):
        head += f'<link rel="stylesheet" property="stylesheet" href="{webpath(css)}" />\n'

    return head

def render_ui():
    js = javascript_html()
    css = css_html()

    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{js}</head>'.encode("utf8"))
        res.body = res.body.replace(b'</body>', f'{css}</body>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response