import random
import hashlib
import json
import requests
import time
import numpy as np
from .translator_base import BaseConfig, BaseTranslator
from .language_code import Lang, LangName

from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

class YoudaoWebConfig(BaseConfig):
    base_url: str = "https://dict.youdao.com/webtranslate"
    key_url: str = "https://dict.youdao.com/webtranslate/key"
    rlog_url: str = "https://rlogs.youdao.com/rlog.php"
    user_id: str = "-762478504@139.226.59.91"
    key: str = "fsdsogkndfokasodnaso"

class BufferFromb64Decoder:
    def __init__(self):
        self.i = self.initI()

    def initI(self):
        s = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        r = [None] * len(s)
        i = [None] * 123
        for a in range(len(s)):
            r[a] = s[a],
            i[ord(s[a])] = a
        i[ord("-")] = 62
        i[ord("_")] = 63
        return i

    def h(self, t: str):
        e = len(t)
        if e % 4 > 0:
            raise Exception("Invalid string. Length must be a multiple of 4")
        n = t.index("=") if "=" in t else -1
        if n == -1:
            n = e
        if n == e:
            r = 0
        else:
            r = 4 - n % 4
        return [n, r]

    def c(self, t, e, n):
        return int(3 * (e + n) / 4 - n)

    def deocde(self, t):
        r = self.h(t)
        s = r[0]
        a = r[1]
        u = np.zeros(self.c(t, s, a), dtype="uint8")
        f = 0
        if a > 0:
            l = s - 4
        else:
            l = s
        n = 0
        while n < l:
            e = self.i[ord(t[n])] << 18 | self.i[ord(t[n + 1])] << 12 | self.i[ord(t[n + 2])] << 6 | self.i[ord(t[n + 3])]
            u[f] = e >> 16 & 255
            f += 1
            u[f] = e >> 8 & 255
            f += 1
            u[f] = 255 & e
            f += 1
            n += 4
        if a == 2:
            e = self.i[ord(t[n])] << 2 | self.i[ord(t[n + 1])] >> 4
            u[f] = 255 & e
            f += 1
        if a == 1:
            e = self.i[ord(t[n])] << 10 | self.i[ord(t[n + 1])] << 4 | self.i[ord(t[n + 2])] >> 2
            u[f] = e >> 8 & 255
            f += 1
            u[f] = 255 & e
            f += 1
        return u

class YouDaoDecoder:
    def __init__(self):
        self.decode_key = "ydsecret://query/key/B*RGygVywfNBwpmBaZg*WT7SIOUP2T0C9WHMZN39j^DAdaZhAnxvGcCY6VYFwnHl"
        self.decode_iv = "ydsecret://query/iv/C@lZe2YzHtZ2CYgaXKSVfsb7Y4QWHjITPPZ0nQp87fBeJ!Iv6v^6fvi2WN@bYpJ4"
        self.b64Decoder = BufferFromb64Decoder()

    def getMD5(self, text: str):
        return hashlib.md5(text.encode("UTF-8"))

    def toUint8(self, text):
        dig = self.getMD5(text).digest()
        byte = bytearray(dig)
        return np.frombuffer(byte, dtype="uint8")

    def decode(self, text):
        key = self.toUint8(self.decode_key).tobytes()
        iv = self.toUint8(self.decode_iv).tobytes()
        text = self.b64Decoder.deocde(text)
        # print(text.tolist())
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypt = cipher.decrypt(text.tobytes())
        # print(decrypt.decode())
        data = unpad(decrypt, AES.block_size).decode("utf-8")
        return data

class YoudaoWebTranslator(BaseTranslator):
    def __init__(self, config: YoudaoWebConfig=YoudaoWebConfig(), hot=True):
        super().__init__(config)
        self.decoder = YouDaoDecoder()
        self.cookies: requests.utils.RequestsCookieJar = requests.utils.cookiejar_from_dict({})
        self.headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "DNT": "1",
            "Pragma": "no-cache",
            "Referer": "https://fanyi.youdao.com/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.58",
        }
        if hot == True:
            self.cookies.set("OUTFOX_SEARCH_USER_ID", self.config.user_id)
            # self.key = "fsdsogkndfokasodnaso"
        else:
            self.getUserID()
            self.getKey()

    def getUserID(self):
        noco = str(2147483647 * random.random())
        params = {
            "_npid": "fanyiweb",
            "_ncat": "pageview",
            "_ncoo": f"{noco}",
            "_nssn": "NULL",
            "_nver": "1.2.0",
            "_ntms": f"{int(time.time() * 1000)}",
            "_nref": "http://fanyi.youdao.com/",
            "_nurl": "https://fanyi.youdao.com/index.html#/",
            "_nres": "1536x864",
            "_nlmf": "1682590851",
            "_njve": "0",
            "_nchr": "utf-8",
            "_nfrg": "/",
            "/": "NULL",
            "screen": "1536*864",
        }
        self.cookies.set("OUTFOX_SEARCH_USER_ID_NCOO", noco)
        res = requests.get(self.config.rlog_url, params=params, headers=self.headers, cookies=self.cookies)
        cookies = res.cookies.get_dict()
        for i in cookies:
            if i == "OUTFOX_SEARCH_USER_ID":
                user_id = cookies["OUTFOX_SEARCH_USER_ID"]
                print(user_id)
                self.cookies.set(i, user_id)

    def getKey(self):
        tp = int(time.time() * 1000)
        string = f"client=fanyideskweb&mysticTime={tp}&product=webfanyi&key=asdjnjfenknafdfsdfsd"
        sign = hashlib.md5(string.encode()).hexdigest()
        params = {
            "keyid": "webfanyi-key-getter",
            "sign": f"{sign}",
            "client": "fanyideskweb",
            "product": "webfanyi",
            "appVersion": "1.0.0",
            "vendor": "web",
            "pointParam": "client,mysticTime,product",
            "mysticTime": f"{tp}",
            "keyfrom": "fanyi.web",
        }
        url = f"?keyid=webfanyi-key-getter&sign={sign}&client=fanyideskweb&product=webfanyi&appVersion=1.0.0&vendor=web&pointParam=client,mysticTime,product&mysticTime={tp}&keyfrom=fanyi.web"
        res = requests.get(self.config.key_url + url, headers=self.headers, cookies=self.cookies)
        js = res.json()
        if js["msg"] == "OK":
            secretKey = js["data"]["secretKey"]
            print(secretKey)
            self.key = secretKey
        else:
            raise Exception("key lose")

    def translate(  self,
                    text: str = "Hello world.",
                    from_lang: Lang = None,
                    to_lang: Lang = Lang.EN,
                    **kwargs ) -> tuple[LangName, LangName, list[tuple[str, str]]]:
        tp = int(time.time() * 1000)
        string = f"client=fanyideskweb&mysticTime={tp}&product=webfanyi&key={self.config.key}"
        sign = hashlib.md5(string.encode()).hexdigest()
        params = {
            "i": text,
            "from": self.get_lang_code(from_lang),
            "to": self.get_lang_code(to_lang),
            "domain": "0",
            "dictResult": "true",
            "keyid": "webfanyi",
            "sign": sign,
            "client": "fanyideskweb",
            "product": "webfanyi",
            "appVersion": "1.0.0",
            "vendor": "web",
            "pointParam": "client,mysticTime,product",
            "mysticTime": str(tp),
            "keyfrom": "fanyi.web",
        }
        res = self.request_url(self.config.base_url, params=params, headers=self.headers, cookies=self.cookies, method="POST")
        
        if res is not None:
            res = json.loads(self.decoder.decode(res))
            # print(res)

            if res["code"] == 0:
                from_code, to_code = res["type"].split("2")
                from_lang = self.get_lang(from_code)
                to_lang = self.get_lang(to_code)
                res_text = [[(item["src"], item["tgt"]) for item in line]
                                for line in res.get("translateResult")]

        return from_lang, to_lang, res_text
            
    LANG_MAPPING = {
        Lang.ZH_CN: "zh-CHS",
        Lang.ZH_TW: "zh-CHT",
    }