# https://cloud.tencent.com/document/product/551/15619

import base64
import hashlib
import hmac
import requests
import time
from .translator_base import BaseConfig, BaseTranslator
from .language_code import Lang, LangName

class TencentConfig(BaseConfig):
    base_url: str = "https://tmt.tencentcloudapi.com"
    secret_id: str = ""
    secret_key: str = ""

class TencentTranslator(BaseTranslator):
    def __init__(self, config: TencentConfig=TencentConfig()):
        super().__init__(config)

    def translate(  self,
                    text: str = "Hello world.",
                    from_lang: LangName = None,
                    to_lang: LangName = None,
                    **kwargs ) -> tuple[LangName, LangName, list[tuple[str, str]]]:

        # Create the request parameters.
        translate_endpoint = "tmt.tencentcloudapi.com"

        params = {
            "Action": "TextTranslate",
            "Nonce": 11886,
            "ProjectId": 0,
            "Region": "ap-beijing",
            "SecretId": self.config.secret_id,
            "Source": self.get_lang_code(from_lang),
            "SourceText": text,
            "Target": self.get_lang_code(to_lang),
            "Timestamp": int(time.time()),
            "Version": "2018-03-21",
        }

        s = "GET" + translate_endpoint + "/?"
        query_str = "&".join(
            "%s=%s" % (k, params[k]) for k in sorted(params)
        )
        hmac_str = hmac.new(
            self.config.secret_key.encode("utf8"),
            (s + query_str).encode("utf8"),
            hashlib.sha1,
        ).digest()
        params["Signature"] = base64.b64encode(hmac_str)

        res = self.request_url(self.config.base_url, params=params)

        if res is not None:
            res = res["Response"]

            from_lang = self.get_lang(res["Source"])
            to_lang = self.get_lang(res["Target"])
            res_text = [(text, res["TargetText"])]
        else:
            res_text = ""

        return from_lang, to_lang, res_text

    LANG_MAPPING = {
        Lang.ZH_CN: "zh",
    }