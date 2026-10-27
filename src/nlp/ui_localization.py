"""
UI Internationalization and Localization Engine for ScholarMind AI.
Provides complete application-level translation for navigation, headers,
buttons, cards, placeholders, and metadata in Gujarati, Hindi, and other supported languages.
"""

import streamlit as st
from typing import Optional, Dict

UI_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "gu": {
        # App brand & Sidebar
        "ScholarMind AI": "સ્કોલરમાઇન્ડ AI",
        "Universal Scientific Intelligence": "સાર્વત્રિક વૈજ્ઞાનિક બુદ્ધિમત્તા",
        "Navigation": "નેવિગેશન",
        "Home": "હોમ (Home)",
        "Research Chat": "સંશોધન ચેટ (Research Chat)",
        "Paper Search": "પેપર શોધો (Paper Search)",
        "Paper Summarizer": "પેપર સારાંશકાર (Paper Summarizer)",
        "Concept Explorer": "ખ્યાલ એક્સપ્લોરર (Concept Explorer)",
        "Peer Review": "પીઅર સમીક્ષા (Peer Review)",
        "Chat History": "ચેટ ઇતિહાસ (Chat History)",
        "Saved Papers": "સાચવેલા પેપર્સ (Saved Papers)",
        "Settings": "સેટિંગ્સ (Settings)",
        "Database": "ડેટાબેઝ",
        "Domain": "ડોમેન (Domain)",
        "Version": "આવૃત્તિ",
        "papers indexed": "પેપર્સ અનુક્રમિત",
        "Multi-Domain Universal Science": "મલ્ટી-ડોમેન સાર્વત્રિક વિજ્ઞાન",
        "Language": "ભાષા (Language)",
        "App Language": "એપ્લિકેશન ભાષા (App Language)",
        "Select App Language": "એપ્લિકેશન ભાષા પસંદ કરો",
        
        # Home Page Hero & Cards
        "Welcome to ScholarMind AI": "સ્કોલરમાઇન્ડ AI માં આપનું સ્વાગત છે",
        "Universal Multi-Domain Scientific Research Assistant & Autonomous Peer Review System.": "સાર્વત્રિક મલ્ટી-ડોમેન વૈજ્ઞાનિક સંશોધન સહાયક અને સ્વાયત્ત પીઅર સમીક્ષા સિસ્ટમ.",
        "Ask a question about research papers, concepts, or search for papers...": "સંશોધન પત્રો અથવા વિભાવનાઓ વિશે પ્રશ્ન પૂછો...",
        "Explain Transformers": "ટ્રાન્સફોર્મર્સ સમજાવો (Explain Transformers)",
        "Find latest NLP papers": "નવીનતમ NLP પેપર્સ શોધો (Latest NLP)",
        "Summarize a paper": "પેપરનો સારાંશ આપો (Summarize Paper)",
        "Visualize RAG": "RAG ની કલ્પના કરો (Visualize RAG)",
        "Search Papers": "પેપર્સ શોધો (Search Papers)",
        "Find relevant research papers from arXiv": "પ્રાસંગિક સંશોધન પેપર્સ શોધો",
        "Search Papers ➔": "પેપર્સ શોધો ➔",
        "Get Summaries": "સારાંશ મેળવો (Get Summaries)",
        "Read concise summaries of complex papers": "જટિલ પેપર્સના સંક્ષિપ્ત સારાંશ વાંચો",
        "Get Summaries ➔": "સારાંશ મેળવો ➔",
        "Explore Concepts": "ખ્યાલોનું અન્વેષણ કરો (Explore Concepts)",
        "Understand technical concepts with visualizations": "વિઝ્યુલાઇઝેશન સાથે તકનીકી વિભાવનાઓ સમજો",
        "Explore Concepts ➔": "ખ્યાલોનું અન્વેષણ કરો ➔",
        "Chat with AI": "AI સાથે ચેટ કરો (Chat with AI)",
        "Ask questions and get research-backed answers": "પ્રશ્નો પૂછો અને સંશોધન-આધારિત જવાબો મેળવો",
        "Chat with AI ➔": "AI સાથે ચેટ કરો ➔",
        
        # Headers & Subtitles
        "Universal Research Assistant & Literature Intelligence": "સાર્વત્રિક સંશોધન સહાયક અને સાહિત્ય બુદ્ધિમત્તા",
        "Explore, chat, summarize, and peer-review research across all scientific domains": "તમામ વૈજ્ઞાનિક ક્ષેત્રોમાં સંશોધનનું અન્વેષણ કરો, ચેટ કરો, સારાંશ આપો અને પીઅર-સમીક્ષા કરો",
        "Ask complex scientific questions grounded in peer-reviewed literature": "પીઅર-સમીક્ષા કરેલા સાહિત્ય પર આધારિત જટિલ વૈજ્ઞાનિક પ્રશ્નો પૂછો",
        "Search Research Papers": "સંશોધન પેપર્સ શોધો (Search Papers)",
        "Discover published literature across IEEE, Springer, Elsevier, ACM, Nature, and arXiv with multi-domain filters.": "મલ્ટી-ડોમેન ફિલ્ટર્સ સાથે IEEE, Springer, Elsevier, ACM, Nature અને arXiv પર પ્રકાશિત સાહિત્ય શોધો.",
        "Paper Summarization": "પેપર સારાંશ (Paper Summarizer)",
        "Get concise and structured summaries of research papers.": "સંશોધન પત્રોના સંક્ષિપ્ત અને સંરચિત સારાંશ મેળવો.",
        "Peer Review & Debate Panel": "પીઅર સમીક્ષા અને ચર્ચા પેનલ (Peer Review)",
        "Simulated conference review committee evaluating technical rigor and novelty.": "તકનીકી કઠોરતા અને નવીનતાનું મૂલ્યાંકન કરતી સિમ્યુલેટેડ કોન્ફરન્સ સમીક્ષા સમિતિ.",
        "Concept Explorer": "ખ્યાલ એક્સપ્લોરર (Concept Explorer)",
        "Visualize semantic relationships between AI concepts and papers.": "AI વિભાવનાઓ અને પેપર્સ વચ્ચેના સિમેન્ટીક સંબંધોની કલ્પના કરો.",
        "Chat History": "ચેટ ઇતિહાસ (Chat History)",
        "Access past research conversations and questions.": "ભૂતકાળના સંશોધન વાર્તાલાપ અને પ્રશ્નો ઍક્સેસ કરો.",
        "Platform Settings": "પ્લેટફોર્મ સેટિંગ્સ (Settings)",
        "Configure LLM backend, retrieval thresholds, and search limits.": "LLM બેકએન્ડ, પુનઃપ્રાપ્તિ થ્રેશોલ્ડ અને શોધ મર્યાદા ગોઠવો.",
        "Paper Details": "પેપર વિગતો (Paper Details)",

        # Action Buttons & Badges
        "Search": "શોધો (Search)",
        "+ New Chat": "+ નવી ચેટ (+ New Chat)",
        "New Chat": "+ નવી ચેટ (+ New Chat)",
        "Summarize": "સારાંશ બનાવો (Summarize)",
        "View Details": "વિગતો જુઓ (View Details)",
        "View Paper": "પેપર જુઓ (View Paper)",
        "Bookmark": "બુકમાર્ક (Bookmark)",
        "★ Saved": "★ સાચવેલ (Saved)",
        "☆ Bookmark": "☆ બુકમાર્ક (Bookmark)",
        "Open DOI ↗": "DOI ખોલો ↗",
        "Open Paper ↗": "પેપર ખોલો ↗",
        "Clear History": "ઇતિહાસ સાફ કરો",
        "Download Review": "સમીક્ષા ડાઉનલોડ કરો",
        "Category": "કેટેગરી (Category)",
        "Publication Year": "પ્રકાશન વર્ષ (Publication Year)",
        "Sort by": "ક્રમબદ્ધ કરો (Sort by)",
        "Relevance": "પ્રાસંગિકતા (Relevance)",
        "Year (Newest)": "વર્ષ (નવું)",
        "Year (Oldest)": "વર્ષ (જૂનું)",
        "Title (A-Z)": "શીર્ષક (A-Z)",
        "Showing": "દર્શાવે છે",
        "publications": "પ્રકાશનો",
        "No papers found": "કોઈ પેપર્સ મળ્યા નથી",
        "Try adjusting your search terms or filters.": "તમારી શોધ શબ્દો અથવા ફિલ્ટર્સ સમાયોજિત કરવાનો પ્રયાસ કરો.",
        
        # Summarizer Tabs & Headings
        "Problem Statement": "સમસ્યાનું નિવેદન (Problem)",
        "Proposed Approach": "સૂચિત અભિગમ (Approach)",
        "Key Contributions": "મુખ્ય યોગદાન (Contributions)",
        "Experimental Results": "પ્રાયોગિક પરિણામો (Results)",
        "Limitations & Constraints": "મર્યાદાઓ અને અવરોધો (Limitations)",
        "Real-World Applications": "વાસ્તવિક-વિશ્વ એપ્લિકેશનો (Applications)",
        "Quick Summary": "ઝડપી સારાંશ (Quick Summary)",
        "Key Takeaways": "મુખ્ય મુદ્દાઓ (Key Takeaways)",
        "Academic Citations": "શૈક્ષણિક સંદર્ભો (Citations)",
        
        # Peer Review
        "Methodology Score": "પદ્ધતિ સ્કોર",
        "Literature Coverage Score": "સાહિત્ય કવરેજ સ્કોર",
        "Final Recommendation": "અંતિમ ભલામણ",
        "Strengths": "મુખ્ય શક્તિઓ / ફાયદા",
        "Weaknesses": "નબળાઈઓ / ખામીઓ",
        "Questions for Authors": "લેખકો માટે પ્રશ્નો",
        "Consensus Summary": "સહમતિ સારાંશ",

        # Chat view
        "Type your scientific question, methodology inquiry, or hypothesis...": "તમારો વૈજ્ઞાનિક પ્રશ્ન, પદ્ધતિની તપાસ અથવા પૂર્વધારણા લખો...",
        "Search papers by topic, methodology, keywords, or title (e.g. 'large language models')...": "વિષય, પદ્ધતિ અથવા શીર્ષક દ્વારા પેપર્સ શોધો...",
        "Search any published paper by title, DOI, or topic (e.g. 'Blockchain Smart voting system', 'AlphaFold')...": "કોઈપણ પ્રકાશિત પેપર શોધો (દા.ત. બ્લોકચેન, આલ્ફાફોલ્ડ)...",
        "Ask a research question...": "સંશોધન પ્રશ્ન પૂછો (દા.ત. ટ્રાન્સફોર્મર, ડીપ લર્નિંગ)..."
    },
    "hi": {
        # App brand & Sidebar
        "ScholarMind AI": "स्कॉलरमाइंड AI",
        "Universal Scientific Intelligence": "सार्वभौमिक वैज्ञानिक बुद्धिमत्ता",
        "Navigation": "नेविगेशन",
        "Home": "होम (Home)",
        "Research Chat": "अनुसंधान चैट (Research Chat)",
        "Paper Search": "शोध पत्र खोजें (Paper Search)",
        "Paper Summarizer": "शोध सारांश (Paper Summarizer)",
        "Concept Explorer": "अवधारणा अन्वेषक (Concept Explorer)",
        "Peer Review": "सहकर्मी समीक्षा (Peer Review)",
        "Chat History": "चैट इतिहास (Chat History)",
        "Saved Papers": "सहेजे गए पत्र (Saved Papers)",
        "Settings": "सेटिंग्स (Settings)",
        "Database": "डेटाबेस",
        "Domain": "डोमेन (Domain)",
        "Version": "संस्करण",
        "papers indexed": "शोध पत्र अनुक्रमित",
        "Multi-Domain Universal Science": "मल्टी-डोमेन सार्वभौमिक विज्ञान",
        "Language": "भाषा (Language)",
        "App Language": "एप्लिकेशन भाषा (App Language)",
        "Select App Language": "एप्लिकेशन भाषा चुनें",
        
        # Home Page Hero & Cards
        "Welcome to ScholarMind AI": "स्कॉलरमाइंड AI में आपका स्वागत है",
        "Universal Multi-Domain Scientific Research Assistant & Autonomous Peer Review System.": "सार्वभौमिक मल्टी-डोमेन वैज्ञानिक अनुसंधान सहायक और स्वायत्त सहकर्मी समीक्षा प्रणाली।",
        "Ask a question about research papers, concepts, or search for papers...": "शोध पत्रों या अवधारणाओं के बारे में प्रश्न पूछें...",
        "Explain Transformers": "ट्रांसफॉर्मर समझाएं (Explain Transformers)",
        "Find latest NLP papers": "नवीनतम NLP पत्र खोजें (Latest NLP)",
        "Summarize a paper": "शोध पत्र का सारांश बनाएं (Summarize Paper)",
        "Visualize RAG": "RAG की कल्पना करें (Visualize RAG)",
        "Search Papers": "शोध पत्र खोजें (Search Papers)",
        "Find relevant research papers from arXiv": "प्रासंगिक शोध पत्र खोजें",
        "Search Papers ➔": "शोध पत्र खोजें ➔",
        "Get Summaries": "सारांश प्राप्त करें (Get Summaries)",
        "Read concise summaries of complex papers": "जटिल शोध पत्रों के संक्षिप्त सारांश पढ़ें",
        "Get Summaries ➔": "सारांश प्राप्त करें ➔",
        "Explore Concepts": "अवधारणाओं का अन्वेषण करें (Explore Concepts)",
        "Understand technical concepts with visualizations": "विज़ुअलाइज़ेशन के साथ तकनीकी अवधारणाओं को समझें",
        "Explore Concepts ➔": "अवधारणाओं का अन्वेषण करें ➔",
        "Chat with AI": "AI से चैट करें (Chat with AI)",
        "Ask questions and get research-backed answers": "प्रश्न पूछें और शोध-आधारित उत्तर प्राप्त करें",
        "Chat with AI ➔": "AI से चैट करें ➔",
        
        # Headers & Subtitles
        "Universal Research Assistant & Literature Intelligence": "सार्वभौमिक अनुसंधान सहायक और साहित्य बुद्धिमत्ता",
        "Explore, chat, summarize, and peer-review research across all scientific domains": "सभी वैज्ञानिक क्षेत्रों में शोध का अन्वेषण करें, चैट करें, सारांश बनाएं और सहकर्मी समीक्षा करें",
        "Ask complex scientific questions grounded in peer-reviewed literature": "सहकर्मी-समीक्षित साहित्य पर आधारित जटिल वैज्ञानिक प्रश्न पूछें",
        "Search Research Papers": "शोध पत्र खोजें (Search Papers)",
        "Discover published literature across IEEE, Springer, Elsevier, ACM, Nature, and arXiv with multi-domain filters.": "मल्टी-डोमेन फिल्टर के साथ IEEE, Springer, Elsevier, ACM, Nature और arXiv पर प्रकाशित साहित्य खोजें।",
        "Paper Summarization": "शोध पत्र सारांशीकरण (Paper Summarizer)",
        "Get concise and structured summaries of research papers.": "शोध पत्रों के संक्षिप्त और संरचित सारांश प्राप्त करें।",
        "Peer Review & Debate Panel": "सहकर्मी समीक्षा और बहस पैनल (Peer Review)",
        "Simulated conference review committee evaluating technical rigor and novelty.": "तकनीकी कठोरता और नवीनता का मूल्यांकन करने वाली सिम्युलेटेड सम्मेलन समीक्षा समिति।",
        "Concept Explorer": "अवधारणा अन्वेषक (Concept Explorer)",
        "Visualize semantic relationships between AI concepts and papers.": "AI अवधारणाओं और शोध पत्रों के बीच अर्थ संबंधी संबंधों की कल्पना करें।",
        "Chat History": "चैट इतिहास (Chat History)",
        "Access past research conversations and questions.": "पिछले शोध वार्तालाप और प्रश्नों तक पहुँचें।",
        "Platform Settings": "प्लेटफ़ॉर्म सेटिंग्स (Settings)",
        "Configure LLM backend, retrieval thresholds, and search limits.": "LLM बैकएंड, पुनर्प्राप्ति सीमाएं और खोज सीमाएं कॉन्फ़िगर करें।",
        "Paper Details": "शोध पत्र विवरण (Paper Details)",

        # Action Buttons & Badges
        "Search": "खोजें (Search)",
        "+ New Chat": "+ नई चैट (+ New Chat)",
        "New Chat": "+ नई चैट (+ New Chat)",
        "Summarize": "सारांश बनाएं (Summarize)",
        "View Details": "विवरण देखें (View Details)",
        "View Paper": "शोध पत्र देखें (View Paper)",
        "Bookmark": "बुकमार्क (Bookmark)",
        "★ Saved": "★ सहेजा गया (Saved)",
        "☆ Bookmark": "☆ बुकमार्क (Bookmark)",
        "Open DOI ↗": "DOI खोलें ↗",
        "Open Paper ↗": "शोध पत्र खोलें ↗",
        "Clear History": "इतिहास साफ़ करें",
        "Download Review": "समीक्षा डाउनलोड करें",
        "Category": "श्रेणी (Category)",
        "Publication Year": "प्रकाशन वर्ष (Publication Year)",
        "Sort by": "क्रमबद्ध करें (Sort by)",
        "Relevance": "प्रासंगिकता (Relevance)",
        "Year (Newest)": "वर्ष (नवीनतम)",
        "Year (Oldest)": "वर्ष (पुरातन)",
        "Title (A-Z)": "शीर्षक (A-Z)",
        "Showing": "दिखा रहा है",
        "publications": "प्रकाशन",
        "No papers found": "कोई शोध पत्र नहीं मिला",
        "Try adjusting your search terms or filters.": "अपने खोज शब्दों या फ़िल्टर को समायोजित करने का प्रयास करें।",
        
        # Summarizer Tabs
        "Problem Statement": "समस्या विवरण (Problem)",
        "Proposed Approach": "प्रस्तावित दृष्टिकोण (Approach)",
        "Key Contributions": "प्रमुख योगदान (Contributions)",
        "Experimental Results": "प्रायोगिक परिणाम (Results)",
        "Limitations & Constraints": "सीमाएं और बाधाएं (Limitations)",
        "Real-World Applications": "वास्तविक-विश्व अनुप्रयोग (Applications)",
        "Quick Summary": "त्वरित सारांश (Quick Summary)",
        "Key Takeaways": "प्रमुख निष्कर्ष (Key Takeaways)",
        "Academic Citations": "अकादमिक उद्धरण (Citations)",
        
        # Peer Review
        "Methodology Score": "प्रणाली स्कोर",
        "Literature Coverage Score": "साहित्य कवरेज स्कोर",
        "Final Recommendation": "अंतिम सिफारिश",
        "Strengths": "प्रमुख ताकत / लाभ",
        "Weaknesses": "कमियां / कमजोरियां",
        "Questions for Authors": "लेखकों के लिए प्रश्न",
        "Consensus Summary": "सहमति सारांश",

        # Chat view
        "Type your scientific question, methodology inquiry, or hypothesis...": "अपना वैज्ञानिक प्रश्न, कार्यप्रणाली पूछताछ या परिकल्पना लिखें...",
        "Search papers by topic, methodology, keywords, or title (e.g. 'large language models')...": "विषय, कार्यप्रणाली या शीर्षक से शोध पत्र खोजें...",
        "Search any published paper by title, DOI, or topic (e.g. 'Blockchain Smart voting system', 'AlphaFold')...": "शीर्षक, डीओआई या विषय से शोध पत्र खोजें...",
        "Ask a research question...": "शोध प्रश्न पूछें (उदा. ट्रांसफॉर्मर, डीप लर्निंग)..."
    }
}

LANGUAGE_DISPLAY_NAMES = {
    "en": "English",
    "gu": "Gujarati (ગુજરાતી)",
    "hi": "Hindi (हिन्दी)",
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
    "zh": "Chinese (中文)",
    "ja": "Japanese (日本語)",
    "ru": "Russian (Русский)",
    "ar": "Arabic (العربية)"
}

def get_current_app_lang() -> str:
    """Returns the currently active application language code ('en', 'gu', 'hi', etc.)."""
    try:
        return st.session_state.get("app_language", "en")
    except Exception:
        return "en"

def t(text: str, lang: Optional[str] = None) -> str:
    """
    Translates UI string into active app language.
    If no translation exists, returns original text.
    """
    if not text:
        return text
    active_lang = lang or get_current_app_lang()
    if active_lang == "en" or active_lang not in UI_TRANSLATIONS:
        return text
    return UI_TRANSLATIONS[active_lang].get(text, text)
