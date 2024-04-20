# https://fanyi-api.baidu.com/product/113

import random
import hashlib
from .translator_base import BaseConfig, BaseTranslator
from .language_code import Lang, LangName

class BaiduConfig(BaseConfig):
    base_url: str = "https://fanyi-api.baidu.com/api/trans/vip/translate"
    app_id: str = ""
    app_key: str = ""

class BaiduTranslator(BaseTranslator):
    def __init__(self, config: BaiduConfig=BaiduConfig()):
        super().__init__(config)

    def translate(  self,
                    text: str = "Hello world.",
                    from_lang: LangName = None,
                    to_lang: LangName = None,
                    **kwargs ) -> tuple[LangName, LangName, list[tuple[str, str]]]:

        # Create the request parameters.
        salt = random.randint(32768, 65536)
        data = f"{self.config.app_id}{text}{salt}{self.config.app_key}"
        sign = hashlib.md5(data.encode("utf-8")).hexdigest()
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }

        payload = {
            "appid": self.config.app_id,
            "q": text,
            "from": self.get_lang_code(from_lang),
            "to": self.get_lang_code(to_lang),
            "salt": salt,
            "sign": sign,
        }

        res = self.request_url(
            self.config.base_url, params=payload, headers=headers
        )

        if res is not None:
            from_lang = self.get_lang(res["from"])
            to_lang = self.get_lang(res["to"])
            res_text = [(item["src"], item["dst"]) for item in res["trans_result"]]
        else:
            res_text = ""

        return from_lang, to_lang, res_text

    LANG_MAPPING = {
        Lang.AF: "afr",
        Lang.SQ: "alb",
        Lang.AM: "amh",
        Lang.AR: "ara",
        Lang.HY: "arm",
        Lang.AS: "asm",
        Lang.AY: "aym",
        Lang.AZ: "aze",
        Lang.EU: "baq",
        Lang.BE: "bel",
        Lang.BN: "ben",
        Lang.BHO: "bho",
        Lang.BS: "bos",
        Lang.BG: "bul",
        Lang.CA: "cat",
        Lang.CEB: "ceb",
        Lang.ZH_CN: "zh",
        Lang.ZH_TW: "cht",
        Lang.ZH_HK: "yue",
        Lang.HR: "hrv",
        Lang.CS: "cs",
        Lang.DA: "dan",
        Lang.DV: "div",
        Lang.NL: "nl",
        Lang.EN: "en",
        Lang.EO: "epo",
        Lang.ET: "est",
        Lang.EE: "ewe",
        Lang.TL: "fil",
        Lang.FI: "fin",
        Lang.FR: "fra",
        Lang.FY: "fry",
        Lang.GL: "glg",
        Lang.KA: "geo",
        Lang.DE: "de",
        Lang.EL: "el",
        Lang.GN: "grn",
        Lang.GU: "guj",
        Lang.HT: "ht",
        Lang.HA: "hau",
        Lang.HAW: "haw",
        Lang.IW: "heb",
        Lang.HI: "hi",
        Lang.HMN: "hmn",
        Lang.HU: "hu",
        Lang.IS: "ice",
        Lang.IG: "ibo",
        Lang.ILO: "ilo",
        Lang.ID: "id",
        Lang.GA: "gle",
        Lang.IT: "it",
        Lang.JA: "jp",
        Lang.JW: "jav",
        Lang.KN: "kan",
        Lang.KK: "kaz",
        Lang.KM: "hkm",
        Lang.RW: "kin",
        Lang.GOM: "kok",
        Lang.KO: "kor",
        Lang.KRI: "kri",
        Lang.KU: "kur",
        Lang.CKB: "kur",
        Lang.KY: "kir",
        Lang.LO: "lao",
        Lang.LA: "lat",
        Lang.LV: "lav",
        Lang.LN: "lin",
        Lang.LT: "lit",
        Lang.LG: "lug",
        Lang.LB: "ltz",
        Lang.MK: "mac",
        Lang.MAI: "mai",
        Lang.MG: "mg",
        Lang.MS: "may",
        Lang.ML: "mal",
        Lang.MT: "mlt",
        Lang.MI: "mao",
        Lang.MR: "mar",
        Lang.LUS: "lus",
        Lang.MN: "mn",
        Lang.MY: "bur",
        Lang.NE: "nep",
        Lang.NO: "nor",
        Lang.OR: "ori",
        Lang.OM: "orm",
        Lang.PS: "pus",
        Lang.FA: "per",
        Lang.PL: "pl",
        Lang.PT: "pt",
        Lang.PA: "pan",
        Lang.QU: "que",
        Lang.RO: "rom",
        Lang.RU: "ru",
        Lang.SM: "sm",
        Lang.SA: "san",
        Lang.GD: "gla",
        Lang.NSO: "nso",
        Lang.SR: "srp",
        Lang.ST: "sot",
        Lang.SN: "sna",
        Lang.SD: "snd",
        Lang.SI: "sin",
        Lang.SK: "sk",
        Lang.SL: "slo",
        Lang.SO: "som",
        Lang.ES: "spa",
        Lang.SU: "sun",
        Lang.SW: "swa",
        Lang.SV: "swe",
        Lang.TG: "tgk",
        Lang.TA: "tam",
        Lang.TT: "tat",
        Lang.TE: "tel",
        Lang.TH: "th",
        Lang.TI: "tir",
        Lang.TS: "tso",
        Lang.TR: "tr",
        Lang.TK: "tuk",
        Lang.AK: "aka",
        Lang.UK: "ukr",
        Lang.UR: "urd",
        Lang.UG: "uig",
        Lang.UZ: "uz",
        Lang.VI: "vie",
        Lang.CY: "wel",
        Lang.XH: "xho",
        Lang.YI: "yid",
        Lang.YO: "yor",
        Lang.ZU: "zul",
    }