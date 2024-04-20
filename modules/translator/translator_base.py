import os
import hashlib
import requests
import modules.paths
from abc import ABC, abstractmethod
from .language_code import Lang, LangName
from modules import util

class BaseConfig(ABC):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__.lower().replace("config", "")
        self.load_config()

    def load_config(self):
        config_path = os.path.join(os.path.join("./configs/translator", f"{self.name}.json"))
        if os.path.exists(config_path):
            data = util.load_json(config_path)
            for k, v in data.items():
                if hasattr(self, k):
                    setattr(self, k, v)

class BaseTranslator(ABC):
    LANG_MAPPING = {}

    def __init__(self, config: BaseConfig):
        super().__init__()

        self.name = self.__class__.__name__
        self.config = config
        self.UN_LANG_MAPPING = { v: k for k, v in self.LANG_MAPPING.items() }

    def __call__(  self,
                    text: str = "Hello world.",
                    from_lang: LangName = None,
                    to_lang: LangName = Lang.EN,
                    use_cache = True,
                    **kwargs ) -> tuple[LangName, LangName, list[tuple[str, str]]]:
        
        params_str = f"{self.name}:{from_lang}:{to_lang}:{text}"
        text_hash = hashlib.sha256(params_str.encode()).hexdigest()[:10]
        cache_root = os.path.join(modules.paths.caches_path, "translator_cache")
        cache_path = os.path.join(cache_root, f"{text_hash}.json")

        if use_cache and os.path.exists(cache_path):
            data = util.load_json(cache_path)
            from_lang, to_lang, trans_texts = getattr(Lang, data["from_lang"]), getattr(Lang, data["to_lang"]), data["trans_texts"]
        else:
            from_lang, to_lang, trans_texts = self.translate(text, from_lang=from_lang, to_lang=to_lang, **kwargs)
            if use_cache:
                os.makedirs(cache_root, exist_ok=True)
                util.save_json(cache_path, {
                        "translator": self.name,
                        "from_lang": from_lang.name,
                        "to_lang": to_lang.name,
                        "trans_texts": trans_texts,
                    })

        return from_lang, to_lang, trans_texts

    @abstractmethod
    def translate(  self,
                    text: str = "Hello world.",
                    from_lang: LangName = None,
                    to_lang: LangName = Lang.EN,
                    **kwargs ) -> tuple[LangName, LangName, list[tuple[str, str]]]:
        return NotImplemented("You need to implement the translate method.")

    def request_url(self, url: str, params: dict={}, headers: dict={}, cookies=None, method="GET"):
        try:
            response = requests.request(method, url, params=params, headers=headers, cookies=cookies)
        except Exception as e:
            print(e)
            return None


        if response.status_code != 200:
            print(f"Request failed with status code {response.status_code}: {response.text}")
            return None

        content_type = response.headers.get("Content-Type")
        if "json" in content_type:
            try:
                res = response.json()
            except:
                res = response.text
        else:
            res = response.text

        if not res:
            res = None

        return res

    def get_lang(self, code: str):
        lang = self.UN_LANG_MAPPING.get(code)
        return lang if lang is not None else Lang.from_code(code)

    def get_lang_code(self, lang: LangName, auto_code=Lang.AUTO.value.code):
        return (self.LANG_MAPPING.get(lang)) or lang.value.code if lang is not None else auto_code

