from enum import Enum
from typing import NamedTuple

class LangName(NamedTuple):
    code: str
    name: str
    local: str

class Lang(Enum):
    AUTO = LangName("auto", "Auto", "Auto")
    AF = LangName("af", "Afrikaans", "Afrikaans")
    SQ = LangName("sq", "Albanian", "Shqip")
    AM = LangName("am", "Amharic", "አማርኛ")
    AR = LangName("ar", "Arabic", "العربية")
    HY = LangName("hy", "Armenian", "Հայերեն")
    AS = LangName("as", "Assamese", "অসমীয়া")
    AY = LangName("ay", "Aymara", "Aimara")
    AZ = LangName("az", "Azerbaijani", "Azərbaycan")
    BM = LangName("bm", "Bambara", "Bamanankan")
    EU = LangName("eu", "Basque", "Euskara")
    BE = LangName("be", "Belarusian", "Беларуская")
    BN = LangName("bn", "Bengali", "বাংলা")
    BHO = LangName("bho", "Bhojpuri", "भोजपुरी")
    BS = LangName("bs", "Bosnian", "Bosanski")
    BG = LangName("bg", "Bulgarian", "Български")
    CA = LangName("ca", "Catalan", "Català")
    CEB = LangName("ceb", "Cebuano", "Binisaya")
    NY = LangName("ny", "Chichewa", "Chi-Chewa")
    ZH = LangName("zh", "Chinese (Simplified)", "中文 (简体)")
    ZH_CN = LangName("zh-CN", "Chinese (Simplified)", "中文 (简体)")
    ZH_TW = LangName("zh-TW", "Chinese (Traditional)", "中文 (繁體)")
    ZH_HK = LangName("zh-HK", "Chinese (Traditional)", "中文 (繁體)")
    CO = LangName("co", "Corsican", "Corsu")
    HR = LangName("hr", "Croatian", "Hrvatski")
    CS = LangName("cs", "Czech", "Čeština")
    DA = LangName("da", "Danish", "Dansk")
    DV = LangName("dv", "Divehi", "ދިވެހިބަސް")
    DOI = LangName("doi", "Dogri", "डोगरी")
    NL = LangName("nl", "Dutch", "Nederlands")
    EN = LangName("en", "English", "English")
    EO = LangName("eo", "Esperanto", "Esperanto")
    ET = LangName("et", "Estonian", "Eesti")
    EE = LangName("ee", "Ewe", "Eʋegbe")
    TL = LangName("tl", "Filipino", "Filipino")
    FI = LangName("fi", "Finnish", "Suomi")
    FR = LangName("fr", "French", "Français")
    FY = LangName("fy", "Frisian", "Frysk")
    GL = LangName("gl", "Galician", "Galego")
    KA = LangName("ka", "Georgian", "ქართული")
    DE = LangName("de", "German", "Deutsch")
    EL = LangName("el", "Greek", "Ελληνικά")
    GN = LangName("gn", "Guarani", "Avañe'ẽ")
    GU = LangName("gu", "Gujarati", "ગુજરાતી")
    HT = LangName("ht", "Haitian Creole", "Kreyòl ayisyen")
    HA = LangName("ha", "Hausa", "Hausa")
    HAW = LangName("haw", "Hawaiian", "ʻŌlelo Hawaiʻi")
    IW = LangName("iw", "Hebrew", "עברית")
    HI = LangName("hi", "Hindi", "हिन्दी")
    HMN = LangName("hmn", "Hmong", "Hmoob")
    HU = LangName("hu", "Hungarian", "Magyar")
    IS = LangName("is", "Icelandic", "Íslenska")
    IG = LangName("ig", "Igbo", "Igbo")
    ILO = LangName("ilo", "Ilocano", "Ilokano")
    ID = LangName("id", "Indonesian", "Bahasa Indonesia")
    GA = LangName("ga", "Irish", "Gaeilge")
    IT = LangName("it", "Italian", "Italiano")
    JA = LangName("ja", "Japanese", "日本語")
    JW = LangName("jw", "Javanese", "Basa Jawa")
    KN = LangName("kn", "Kannada", "ಕನ್ನಡ")
    KK = LangName("kk", "Kazakh", "Қазақ")
    KM = LangName("km", "Khmer", "ខ្មែរ")
    RW = LangName("rw", "Kinyarwanda", "Kinyarwanda")
    GOM = LangName("gom", "Konkani", "Konkani")
    KO = LangName("ko", "Korean", "한국어")
    KRI = LangName("kri", "Krio", "Krio")
    KU = LangName("ku", "Kurdish (Kurmanji)", "Kurdî")
    CKB = LangName("ckb", "Kurdish (Sorani)", "کوردی (سۆرانی)")
    KY = LangName("ky", "Kyrgyz", "Кыргызча")
    LO = LangName("lo", "Lao", "ລາວ")
    LA = LangName("la", "Latin", "Latina")
    LV = LangName("lv", "Latvian", "Latviešu")
    LN = LangName("ln", "Lingala", "Lingala")
    LT = LangName("lt", "Lithuanian", "Lietuvių")
    LG = LangName("lg", "Luganda", "Luganda")
    LB = LangName("lb", "Luxembourgish", "Lëtzebuergesch")
    MK = LangName("mk", "Macedonian", "Македонски")
    MAI = LangName("mai", "Maithili", "मैथिली")
    MG = LangName("mg", "Malagasy", "Malagasy")
    MS = LangName("ms", "Malay", "Bahasa Melayu")
    ML = LangName("ml", "Malayalam", "മലയാളം")
    MT = LangName("mt", "Maltese", "Malti")
    MI = LangName("mi", "Maori", "Te Reo Māori")
    MR = LangName("mr", "Marathi", "मराठी")
    MNI_MTEI = LangName("mni_mtei", "Meiteilon (Manipuri)", "ꯃꯤꯇꯦꯡ ꯈꯪꯟꯖꯦꯇꯥ")
    LUS = LangName("lus", "Mizo", "Mizo ṭawng")
    MN = LangName("mn", "Mongolian", "Монгол")
    MY = LangName("my", "Myanmar", "မြန်မာစာ")
    NE = LangName("ne", "Nepali", "नेपाली")
    NO = LangName("no", "Norwegian", "Norsk")
    OR = LangName("or", "Odia (Oriya)", "ଓଡ଼ିଆ")
    OM = LangName("om", "Oromo", "Afaan Oromo")
    PS = LangName("ps", "Pashto", "پښتو")
    FA = LangName("fa", "Persian", "فارسی")
    PL = LangName("pl", "Polish", "Polski")
    PT = LangName("pt", "Portuguese", "Português")
    PA = LangName("pa", "Punjabi", "ਪੰਜਾਬੀ")
    QU = LangName("qu", "Quechua", "Runa Simi")
    RO = LangName("ro", "Romanian", "Română")
    RU = LangName("ru", "Russian", "Русский")
    SM = LangName("sm", "Samoan", "Gagana Samoa")
    SA = LangName("sa", "Sanskrit", "संस्कृतम्")
    GD = LangName("gd", "Scots Gaelic", "Gàidhlig")
    NSO = LangName("nso", "Sepedi", "Sepedi")
    SR = LangName("sr", "Serbian", "Српски")
    ST = LangName("st", "Sesotho", "Sesotho")
    SN = LangName("sn", "Shona", "chiShona")
    SD = LangName("sd", "Sindhi", "سنڌي")
    SI = LangName("si", "Sinhala", "සිංහල")
    SK = LangName("sk", "Slovak", "Slovenčina")
    SL = LangName("sl", "Slovenian", "Slovenščina")
    SO = LangName("so", "Somali", "Af Soomaali")
    ES = LangName("es", "Spanish", "Español")
    SU = LangName("su", "Sundanese", "Basa Sunda")
    SW = LangName("sw", "Swahili", "Kiswahili")
    SV = LangName("sv", "Swedish", "Svenska")
    TG = LangName("tg", "Tajik", "Тоҷикӣ")
    TA = LangName("ta", "Tamil", "தமிழ்")
    TT = LangName("tt", "Tatar", "Татарча")
    TE = LangName("te", "Telugu", "తెలుగు")
    TH = LangName("th", "Thai", "ไทย")
    TI = LangName("ti", "Tigrinya", "ትግርኛ")
    TS = LangName("ts", "Tsonga", "Xitsonga")
    TR = LangName("tr", "Turkish", "Türkçe")
    TK = LangName("tk", "Turkmen", "Türkmençe")
    AK = LangName("ak", "Twi", "Akan")
    UK = LangName("uk", "Ukrainian", "Українська")
    UR = LangName("ur", "Urdu", "اردو")
    UG = LangName("ug", "Uyghur", "ئۇيغۇرچە")
    UZ = LangName("uz", "Uzbek", "O'zbek")
    VI = LangName("vi", "Vietnamese", "Tiếng Việt")
    CY = LangName("cy", "Welsh", "Cymraeg")
    XH = LangName("xh", "Xhosa", "isiXhosa")
    YI = LangName("yi", "Yiddish", "ייִדיש")
    YO = LangName("yo", "Yoruba", "Yorùbá")
    ZU = LangName("zu", "Zulu", "Zulu")

    def from_code(code: str):
        langs = list(filter(lambda x: x.value.code.lower() == code.lower(), Lang))
        return langs[0] if langs else None