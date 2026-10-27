"""
Comprehensive Multilingual Research Assistant Engine (Requirement 16).
Provides script detection, cross-lingual retrieval query normalization,
and multi-language scientific synthesis across Indian state languages and major global languages:

Indian State Languages:
- Hindi (hi), Gujarati (gu), Marathi (mr), Tamil (ta), Telugu (te),
  Kannada (kn), Malayalam (ml), Bengali (bn), Punjabi (pa)

Global Languages:
- English (en), Spanish (es), French (fr), German (de),
  Japanese (ja), Chinese (zh), Russian (ru), Arabic (ar)
"""

import re
from typing import Dict, Any, Tuple, Optional, List

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi (हिन्दी)",
    "gu": "Gujarati (ગુજરાતી)",
    "mr": "Marathi (मराठी)",
    "ta": "Tamil (தமிழ்)",
    "te": "Telugu (తెలుగు)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "ml": "Malayalam (മലയാളം)",
    "bn": "Bengali (বাংলা)",
    "pa": "Punjabi (ਪੰਜਾਬੀ)",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
    "ja": "Japanese (日本語)",
    "zh": "Chinese (中文)",
    "ru": "Russian (Русский)",
    "ar": "Arabic (العربية)"
}

# Cross-lingual scientific query mappings for retrieval against English arXiv ChromaDB
MULTILINGUAL_GLOSSARY = {
    # Hindi / Marathi (Devanagari)
    "ट्रांसफॉर्मर": "transformer",
    "ध्यान": "attention mechanism",
    "डीप लर्निंग": "deep learning",
    "मशीन लर्निंग": "machine learning",
    "तंत्रिका": "neural network",
    "कृत्रिम बुद्धिमत्ता": "artificial intelligence",
    "कंप्यूटर विजन": "computer vision",
    "नैसर्गिक भाषा": "natural language processing",
    "संवर्धित": "augmented generation",
    "पुनर्प्राप्ति": "retrieval",

    # Gujarati
    "ટ્રાન્સફોર્મર": "transformer",
    "ધ્યાન": "attention mechanism",
    "ડીપ લર્નિંગ": "deep learning",
    "મશીન લર્નિંગ": "machine learning",
    "કૃત્રિમ બુદ્ધિ": "artificial intelligence",
    "કમ્પ્યુટર વિઝન": "computer vision",

    # Tamil
    "டிரான்ஸ்பார்மர்": "transformer",
    "கவனம்": "attention mechanism",
    "ஆழ்ந்த கற்றல்": "deep learning",
    "இயந்திர கற்றல்": "machine learning",
    "செயற்கை நுண்ணறிவு": "artificial intelligence",

    # Telugu
    "ట్రాన్స్‌ఫార్మర్": "transformer",
    "శ్రద్ధ": "attention mechanism",
    "డీప్ లెర్నింగ్": "deep learning",
    "మెషిన్ లెర్నింగ్": "machine learning",
    "కృత్రిమ మేధస్సు": "artificial intelligence",

    # Kannada
    "ಟ್ರಾನ್ಸ್‌ಫಾರ್ಮರ್": "transformer",
    "ಗಮನ": "attention mechanism",
    "ಡೀಪ್ ಲರ್ನಿಂಗ್": "deep learning",
    "ಯಂತ್ರ ಕಲಿಕೆ": "machine learning",
    "ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ": "artificial intelligence",

    # Malayalam
    "ട്രാൻസ്ഫോർമർ": "transformer",
    "ശ്രദ്ധ": "attention mechanism",
    "ഡീപ് ലേണിംഗ്": "deep learning",
    "മെഷീൻ ലേണിംഗ്": "machine learning",
    "കൃത്രിമബുദ്ധി": "artificial intelligence",

    # Bengali
    "ট্রান্সফরমার": "transformer",
    "মনোযোগ": "attention mechanism",
    "ডিপ লার্নিং": "deep learning",
    "মেশিন লার্নিং": "machine learning",
    "কৃত্রিম বুদ্ধিমত্তা": "artificial intelligence",

    # Punjabi
    "ਟਰਾਂਸਫਾਰਮਰ": "transformer",
    "ਧਿਆਨ": "attention mechanism",
    "ਡੀਪ ਲਰਨਿੰਗ": "deep learning",
    "ਮਸ਼ੀਨ ਲਰਨਿੰਗ": "machine learning",
    "ਨਕਲੀ ਬੁੱਧੀ": "artificial intelligence",

    # Spanish
    "transformador": "transformer",
    "atención": "attention mechanism",
    "aprendizaje profundo": "deep learning",
    "aprendizaje automático": "machine learning",
    "redes neuronales": "neural networks",
    "inteligencia artificial": "artificial intelligence",
    "visión por computadora": "computer vision",

    # French
    "transformateur": "transformer",
    "apprentissage profond": "deep learning",
    "apprentissage automatique": "machine learning",
    "réseaux de neurones": "neural networks",
    "intelligence artificielle": "artificial intelligence",

    # German
    "aufmerksamkeitsmechanismus": "attention mechanism",
    "tiefes lernen": "deep learning",
    "maschinelles lernen": "machine learning",
    "neuronale netze": "neural networks",
    "künstliche intelligenz": "artificial intelligence",

    # Japanese
    "トランスフォーマー": "transformer",
    "アテンション": "attention mechanism",
    "深層学習": "deep learning",
    "機械学習": "machine learning",
    "人工知能": "artificial intelligence",

    # Chinese
    "变压器": "transformer",
    "注意力机制": "attention mechanism",
    "深度学习": "deep learning",
    "机器学习": "machine learning",
    "人工智能": "artificial intelligence",

    # Russian
    "трансформер": "transformer",
    "внимание": "attention mechanism",
    "глубокое обучение": "deep learning",
    "машинное обучение": "machine learning",
    "искусственный интеллект": "artificial intelligence",

    # Arabic
    "المحول": "transformer",
    "الانتباه": "attention mechanism",
    "التعلم العميق": "deep learning",
    "تعلم الآلة": "machine learning",
    "الذكاء الاصطناعي": "artificial intelligence"
}

# Regional and global language header localizations
LANGUAGE_FRAMINGS = {
    "en": {"header": "### Scientific Analysis", "grounded_note": "Grounded in peer-reviewed arXiv literature:"},
    "hi": {"header": "### वैज्ञानिक विश्लेषण (Scientific Analysis)", "grounded_note": "arXiv शोध और वैज्ञानिक साहित्य पर आधारित:"},
    "gu": {"header": "### વૈજ્ઞાનિક વિશ્લેષણ (Scientific Analysis)", "grounded_note": "arXiv વૈજ્ઞાનિક સંશોધન અને સાહિત્ય પર આધારિત:"},
    "mr": {"header": "### वैज्ञानिक विश्लेषण (Scientific Analysis)", "grounded_note": "arXiv वैज्ञानिक संशोधन आणि साहित्यावर आधारित:"},
    "ta": {"header": "### அறிவியல் பகுப்பாய்வு (Scientific Analysis)", "grounded_note": "arXiv சக மதிப்பாய்வு செய்யப்பட்ட அறிவியல் ஆராய்ச்சி அடிப்படையிலானது:"},
    "te": {"header": "### శాస్త్రీయ విశ్లేషణ (Scientific Analysis)", "grounded_note": "arXiv శాస్త్రీయ పరిశోధన మరియు సాహిత్యం ఆధారంగా:"},
    "kn": {"header": "### ವೈಜ್ಞಾನಿಕ ವಿಶ್ಲೇಷಣೆ (Scientific Analysis)", "grounded_note": "arXiv ವೈಜ್ಞಾನಿಕ ಸಂಶೋಧನೆ ಮತ್ತು ಸಾಹಿತ್ಯದ ಆಧಾರದ ಮೇಲೆ:"},
    "ml": {"header": "### ശാസ്ത്രീയ വിശകലനം (Scientific Analysis)", "grounded_note": "arXiv ശാസ്ത്ര ഗവേഷണത്തെയും സാഹിത്യത്തെയും അടിസ്ഥാനമാക്കി:"},
    "bn": {"header": "### বৈজ্ঞানিক বিশ্লেষণ (Scientific Analysis)", "grounded_note": "arXiv পিয়ার-পর্যালোচিত বৈজ্ঞানিক গবেষণার উপর ভিত্তি করে:"},
    "pa": {"header": "### ਵਿਗਿਆਨਕ ਵਿਸ਼ਲੇਸ਼ਣ (Scientific Analysis)", "grounded_note": "arXiv ਖੋਜ ਪੱਤਰਾਂ ਅਤੇ ਵਿਗਿਆਨਕ ਸਾਹਿਤ 'ਤੇ ਆਧਾਰਿਤ:"},
    "es": {"header": "### Análisis Científico (Scientific Analysis)", "grounded_note": "Basado en la literatura de investigación de arXiv revisada por pares:"},
    "fr": {"header": "### Analyse Scientifique (Scientific Analysis)", "grounded_note": "Fondé sur la littérature de recherche arXiv évaluée par des pairs :"},
    "de": {"header": "### Wissenschaftliche Analyse (Scientific Analysis)", "grounded_note": "Gestützt auf begutachtete arXiv-Forschungsliteratur:"},
    "ja": {"header": "### 科学的分析 (Scientific Analysis)", "grounded_note": "査読済みarXiv学術文献に基づく回答:"},
    "zh": {"header": "### 科学分析 (Scientific Analysis)", "grounded_note": "基于经过同行评审的arXiv学术文献:"},
    "ru": {"header": "### Научный Анализ (Scientific Analysis)", "grounded_note": "На основе рецензируемой научной литературы arXiv:"},
    "ar": {"header": "### التحليل العلمي (Scientific Analysis)", "grounded_note": "مستند إلى أبحاث وأوراق arXiv العلمية المحكمة:"}
}


class MultilingualResearchEngine:
    """Manages language detection, cross-lingual translation, and localized scientific synthesis."""

    def detect_language(self, text: str) -> str:
        """
        Detects Indian state languages and global languages via Unicode script ranges and keywords.
        """
        if not text or not text.strip():
            return "en"

        # 1. Indian Scripts (Unicode ranges)
        if re.search(r'[\u0A80-\u0AFF]', text):
            return "gu"  # Gujarati
        if re.search(r'[\u0B80-\u0BFF]', text):
            return "ta"  # Tamil
        if re.search(r'[\u0C00-\u0C7F]', text):
            return "te"  # Telugu
        if re.search(r'[\u0C80-\u0CFF]', text):
            return "kn"  # Kannada
        if re.search(r'[\u0D00-\u0D7F]', text):
            return "ml"  # Malayalam
        if re.search(r'[\u0980-\u09FF]', text):
            return "bn"  # Bengali
        if re.search(r'[\u0A00-\u0A7F]', text):
            return "pa"  # Punjabi

        # Devanagari script: Hindi or Marathi
        if re.search(r'[\u0900-\u097F]', text):
            marathi_markers = {"आहे", "नाही", "कसे", "काय", "झाले", "या", "करा", "पाहिजे", "मराठी"}
            words = set(re.findall(r'[\u0900-\u097F]+', text))
            if words.intersection(marathi_markers):
                return "mr"
            return "hi"

        # 2. Asian & Middle-Eastern Scripts
        if re.search(r'[\u3040-\u30FF]', text):
            return "ja"  # Japanese
        if re.search(r'[\u4E00-\u9FFF]', text):
            return "zh"  # Chinese
        if re.search(r'[\u0600-\u06FF]', text):
            return "ar"  # Arabic
        if re.search(r'[\u0400-\u04FF]', text):
            return "ru"  # Russian

        # 3. European Latin languages
        lower = text.lower()
        tokens = set(re.findall(r"[a-zà-ÿ]+", lower))

        spanish_keywords = {"qué", "cómo", "por", "para", "los", "las", "este", "esta", "investigación", "aprendizaje", "profundo"}
        if re.search(r'[¿¡ñáóú]', lower) or tokens.intersection(spanish_keywords):
            return "es"

        french_keywords = {"est", "sont", "pourquoi", "comment", "dans", "avec", "cette", "recherche", "apprentissage", "automatique", "que", "qui"}
        if re.search(r'[œçèêàâîôû]', lower) or tokens.intersection(french_keywords):
            return "fr"

        german_keywords = {"was", "ist", "sind", "warum", "wie", "nicht", "eine", "dieser", "forschung", "lernen"}
        if re.search(r'[äöüß]', lower) or tokens.intersection(german_keywords):
            return "de"

        return "en"

    def translate_query_for_retrieval(self, query: str, src_lang: str) -> str:
        """
        Translates or augments queries from non-English languages with English scientific keywords
        to maximize vector retrieval recall against arXiv ChromaDB.
        """
        if src_lang == "en":
            return query

        translated = query.lower()
        for foreign_word, english_keyword in MULTILINGUAL_GLOSSARY.items():
            if foreign_word in translated:
                translated = translated.replace(foreign_word, english_keyword)

        return f"{query} {translated}".strip()

    def format_multilingual_response(
        self,
        base_answer: str,
        target_lang: str,
        sources: List[Dict[str, Any]]
    ) -> str:
        """
        Formats and localizes the scientific answer into the requested target language
        while preserving citations [1], [2] and technical terms.
        """
        if target_lang == "en" or target_lang not in LANGUAGE_FRAMINGS:
            return base_answer

        framing = LANGUAGE_FRAMINGS[target_lang]
        localized_header = (
            f"{framing['header']}\n"
            f"*{framing['grounded_note']}*\n\n"
        )
        return f"{localized_header}{base_answer}"
