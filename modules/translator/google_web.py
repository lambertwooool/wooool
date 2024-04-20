from bs4 import BeautifulSoup
from .translator_base import BaseConfig, BaseTranslator
from .language_code import Lang, LangName

class GoogleWebConfig(BaseConfig):
    base_url: str = "https://translate.google.com/m"
    element_tag: str = "div"
    element_query: dict = {"class": "result-container"}

class GoogleWebTranslator(BaseTranslator):
    def __init__(self, config: GoogleWebConfig=GoogleWebConfig()):
        super().__init__(config)

    def translate(  self,
                    text: str = "Hello world.",
                    from_lang: Lang = None,
                    to_lang: Lang = Lang.EN,
                    **kwargs ) -> tuple[LangName, LangName, list[tuple[str, str]]]:

        params = {
            "sl": self.get_lang_code(from_lang),
            "tl": self.get_lang_code(to_lang),
            "q": text,
        }

        res = self.request_url(self.config.base_url, params=params)

        if res is not None:
            soup = BeautifulSoup(res, "html.parser")
            
            body = soup.find("body")
            trans_text = body.find(self.config.element_tag, self.config.element_query).text
            res_text = [[(text, trans_text)]]
            from_lang = from_lang if from_lang is not None else Lang.AUTO

        return from_lang, to_lang, res_text