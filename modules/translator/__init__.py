import torch
from .translator_base import BaseConfig, BaseTranslator
from .baidu import BaiduTranslator, BaiduConfig
from .tencent import TencentTranslator, TencentConfig
from .deepl import DeepLTranslator, DeepLConfig, DeepLProConfig
from .youdao_web import YoudaoWebTranslator, YoudaoWebConfig
from .google_web import GoogleWebTranslator, GoogleWebConfig
from .language_code import Lang, LangName

__all__ = [
    "BaiduTranslator",
    "BaiduConfig",
    "TencentTranslator",
    "TencentConfig",
    "DeepLTranslator",
    "DeepLConfig",
    "DeepLProConfig",
    "YoudaoWebTranslator",
    "YoudaoWebConfig",
    "GoogleWebTranslator",
    "GoogleWebConfig",

    "Lang",
    "LangName",
]

MODELS = {
    'google_web': { 'class': GoogleWebTranslator },
    'baidu': { 'class': BaiduTranslator },
    'tencent': { 'class': TencentTranslator },
    'deepl': { 'class': DeepLTranslator },
    'youdao_web': { 'class': YoudaoWebTranslator },
}

class TranslatorProcessor:
    def __init__(self, processor_id: str) -> None:
        """Processor that can be used to translate text processors

        Args:
            processor_id (str)
        """
        if processor_id not in MODELS:
            raise ValueError(f"{processor_id} is not a valid processor id. Please make sure to choose one of {', '.join(MODELS.keys())}")

        self.processor_id = processor_id
        self.processor: BaseTranslator = None

    def model_keys():
        return MODELS.keys()

    def load_processor(self):
        """Load processor

        """
        self.processor = MODELS[self.processor_id]['class']()
    
    @torch.inference_mode()
    def __call__(   self,
                    text: str,
                    from_lang: LangName = None,
                    to_lang: LangName = None,
                    **kwargs) -> str:
                            
        if text is None or text.strip() == "":
            return ""

        if self.processor is None:
            self.load_processor()
        
        # from_lang, to_lang, trans_texts = self.processor.translate(text, from_lang=from_lang, to_lang=to_lang, **kwargs)
        from_lang, to_lang, trans_texts = self.processor(text, from_lang=from_lang, to_lang=to_lang, use_cache=True, **kwargs)
        trans_text = ''.join([t_text[1] for t_line in trans_texts for t_text in t_line])
        
        return trans_text