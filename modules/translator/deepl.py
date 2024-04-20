# https://developers.deepl.com/docs/api-reference/translate

import random
import hashlib
from .translator_base import BaseConfig, BaseTranslator
from .language_code import Lang, LangName

class DeepLConfig(BaseConfig):
    base_url: str = "https://api-free.deepl.com/v2/"
    app_key: str = ""

class DeepLProConfig(DeepLConfig):
    base_url: str = "https://api.deepl.com/v2/"
    app_key: str = ""

class DeepLTranslator(BaseTranslator):
    def __init__(self, config: DeepLConfig=DeepLConfig()):
        super().__init__(config)

    def translate(  self,
                    text: str = "Hello world.",
                    from_lang: LangName = None,
                    to_lang: LangName = None,
                    **kwargs ) -> tuple[LangName, LangName, list[tuple[str, str]]]:

        # Create the request parameters.
        translate_endpoint = "translate"
        source_lang = self.get_lang_code(from_lang, auto_code="")
        params = {
            "auth_key": self.config.app_key,
            "source_lang": source_lang,
            "target_lang": self.get_lang_code(to_lang),
            "text": text,
        }

        res = self.request_url(
            f"{self.config.base_url}/translate", params=params
        )

        if res is not None:
            res = res["translations"][0]

            from_lang = self.get_lang(source_lang or res["detected_source_language"])
            res_text = [[(text, res["text"])]]
        else:
            res_text = ""

        return from_lang, to_lang, res_text

    LANG_MAPPING = {
        Lang.ZH_CN: "zh",
    }