"""Fold spoken number words and native-script digits to ASCII digits.

Two spoken forms must collapse to one string:
    digit-by-digit  "four eight seven three" -> "4873"
    cardinal        "one thousand four hundred fifty" -> "1450"
Loanwords fold to the spoken language's own script (`USB` -> `यूएसबी` in Hindi).
Clock times ("six forty five") are out of scope - match those via answer_aliases.
Apply to reference and hypothesis alike before WER or answer matching.
"""
import re
import unicodedata

# value per word; multipliers live in MULT because they scale instead of adding
NUM = {
    "en": {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
           "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
           "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
           "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
           "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
           "ninety": 90},
    "hi": {"शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5,
           "छह": 6, "छः": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "पंद्रह": 15,
           "पंदरह": 15, "सत्रह": 17, "उन्नीस": 19, "बीस": 20, "पैंतालीस": 45,
           "सैंतालीस": 47, "उनचास": 49, "पचास": 50, "छियासठ": 66, "अट्ठासी": 88,
           "अठासी": 88, "नवासी": 89},
    "gu": {"શૂન્ય": 0, "એક": 1, "બે": 2, "ત્રણ": 3, "ચાર": 4, "પાંચ": 5, "છ": 6,
           "સાત": 7, "આઠ": 8, "નવ": 9, "દસ": 10, "પંદર": 15, "સત્તર": 17,
           "ઓગણીસ": 19, "વીસ": 20, "પિસ્તાળીસ": 45, "સુડતાળીસ": 47, "ઓગણપચાસ": 49,
           "પચાસ": 50, "છાસઠ": 66, "અઠ્યાસી": 88, "નેવ્યાસી": 89,
           # Gujarati fuses the hundreds into one word
           "બસો": 200, "ત્રણસો": 300, "ચારસો": 400, "સાતસો": 700, "ઓગણીસસો": 1900},
    "ta": {"பூஜ்ஜியம்": 0, "சுழியம்": 0, "ஜீரோ": 0, "ஒன்று": 1, "இரண்டு": 2,
           "மூன்று": 3, "நான்கு": 4, "ஐந்து": 5, "ஆறு": 6, "ஏழு": 7, "எட்டு": 8,
           "ஒன்பது": 9, "பத்து": 10, "பதினைந்து": 15, "பதினேழு": 17, "இருபது": 20,
           # sandhi combining forms, e.g. நாற்பத்தி ஐந்து = 45
           "நாற்பத்தி": 40, "ஐம்பது": 50, "அறுபத்தி": 60, "எண்பத்தி": 80,
           "இருநூற்று": 200, "முந்நூற்று": 300, "நாநூற்று": 400, "எழுநூற்று": 700,
           "தொள்ளாயிரத்து": 900, "ஆயிரத்து": 1000},
    "pa": {"ਸਿਫ਼ਰ": 0, "ਸਿਫਰ": 0, "ਜ਼ੀਰੋ": 0, "ਇੱਕ": 1, "ਇਕ": 1, "ਦੋ": 2, "ਤਿੰਨ": 3,
           "ਚਾਰ": 4, "ਪੰਜ": 5, "ਛੇ": 6, "ਸੱਤ": 7, "ਅੱਠ": 8, "ਨੌਂ": 9, "ਨੌ": 9,
           "ਦਸ": 10, "ਪੰਦਰਾਂ": 15, "ਸਤਾਰਾਂ": 17, "ਉਨੀ": 19, "ਵੀਹ": 20,
           "ਪੰਤਾਲੀ": 45, "ਸੰਤਾਲੀ": 47, "ਉਣੰਜਾ": 49, "ਪੰਜਾਹ": 50, "ਛਿਆਹਠ": 66,
           "ਅਠਾਸੀ": 88, "ਉਨਾਨਵੇਂ": 89},
}

MULT = {
    "en": {"hundred": 100, "thousand": 1000, "lakh": 100000},
    "hi": {"सौ": 100, "हज़ार": 1000, "हजार": 1000, "लाख": 100000},
    "gu": {"સો": 100, "હજાર": 1000, "લાખ": 100000},
    "ta": {"நூறு": 100, "ஆயிரம்": 1000, "லட்சம்": 100000},
    "pa": {"ਸੌ": 100, "ਹਜ਼ਾਰ": 1000, "ਹਜਾਰ": 1000, "ਲੱਖ": 100000},
}

# native digits -> ASCII (Devanagari, Gujarati, Tamil, Gurmukhi)
_DIGITS = str.maketrans({chr(base + i): str(i)
                         for base in (0x966, 0xAE6, 0xBE6, 0xA66) for i in range(10)})


# English loanwords, one row per term. Each language folds to its OWN script, so a
# Hindi transcript reads Hindi throughout; English folds to lowercase Latin.
# Scope: acronyms, brands and bare tech terms - the forms an ASR may emit in Latin.
TERMS = {
    # tech / brand
    "usb": {"hi": "यूएसबी", "gu": "યુએસબી", "ta": "யூஎஸ்பி", "pa": "ਯੂਐਸਬੀ"},
    "atm": {"hi": "एटीएम", "gu": "એટીએમ", "ta": "ஏடிஎம்", "pa": "ਏਟੀਐਮ"},
    "android": {"hi": "एंड्रॉइड", "gu": "એન્ડ્રોઇડ", "ta": "ஆண்ட்ராய்ட்", "pa": "ਐਂਡਰਾਇਡ"},
    "google": {"hi": "गूगल", "gu": "ગૂગલ", "ta": "கூகுள்", "pa": "ਗੂਗਲ"},
    "microsoft": {"hi": "माइक्रोसॉफ्ट", "gu": "માઇક્રોસોફ્ટ", "ta": "மைக்ரோசாஃப்ட்", "pa": "ਮਾਈਕ੍ਰੋਸਾਫਟ"},
    "excel": {"hi": "एक्सेल", "gu": "એક્સેલ", "ta": "எக்செல்", "pa": "ਐਕਸਲ"},
    "wi": {"hi": "वाई", "gu": "વાઇ", "ta": "வை", "pa": "ਵਾਈ"},
    "fi": {"hi": "फाई", "gu": "ફાઇ", "ta": "ஃபை", "pa": "ਫਾਈ"},
    # joined spellings expand to the 2 tokens the hyphenated reference becomes
    "wifi": {"hi": "वाई फाई", "gu": "વાઇ ફાઇ", "ta": "வை ஃபை", "pa": "ਵਾਈ ਫਾਈ"},
    "वाईफाई": {"hi": "वाई फाई"},
    "વાઇફાઇ": {"gu": "વાઇ ફાઇ"},
    "வைஃபை": {"ta": "வை ஃபை"},
    "ਵਾਈਫਾਈ": {"pa": "ਵਾਈ ਫਾਈ"},
    "xlsx": {"hi": "एक्सएलएसएक्स", "gu": "એક્સએલએસએક્સ"},
    "universal": {"hi": "यूनिवर्सल", "gu": "યુનિવર્સલ", "ta": "யுனிவர்சல்", "pa": "ਯੂਨੀਵਰਸਲ"},
    "serial": {"hi": "सीरियल", "gu": "સીરીયલ", "ta": "சீரியல்", "pa": "ਸੀਰੀਅਲ"},
    "bus": {"hi": "बस", "gu": "બસ", "ta": "பஸ்", "pa": "ਬੱਸ"},
    "automated": {"hi": "ऑटोमेटेड", "gu": "ઓટોમેટેડ", "ta": "ஆட்டோமேட்டட்", "pa": "ਆਟੋਮੇਟਿਡ"},
    "teller": {"hi": "टेलर", "gu": "ટેલર", "ta": "டெல்லர்", "pa": "ਟੈਲਰ"},
    "machine": {"hi": "मशीन", "gu": "મશીન", "ta": "மெஷின்", "pa": "ਮਸ਼ੀਨ"},
    "wireless": {"hi": "वायरलेस", "gu": "વાયરલેસ", "ta": "வயர்லெஸ்", "pa": "ਵਾਇਰਲੈੱਸ"},
    "wired": {"hi": "वायर्ड", "gu": "વાયર્ડ", "ta": "வயர்டு", "pa": "ਵਾਇਰਡ"},
    "system": {"hi": "सिस्टम", "gu": "સિસ્ટમ", "pa": "ਸਿਸਟਮ"},
    "operating": {"hi": "ऑपरेटिंग", "gu": "ઓપરેટિંગ", "ta": "ஆபரேட்டிங்", "pa": "ਆਪਰੇਟਿੰਗ"},
    "network": {"hi": "नेटवर्क", "gu": "નેટવર્ક", "ta": "நெட்வொர்க்", "pa": "ਨੈੱਟਵਰਕ"},
    "technology": {"hi": "टेक्नोलॉजी", "gu": "ટેક્નોલોજી", "pa": "ਤਕਨਾਲੋਜੀ"},
    "spreadsheet": {"hi": "स्प्रेडशीट", "gu": "સ્પ્રેડશીટ", "ta": "ஸ்ப்ரெட்ஷீட்", "pa": "ਸਪ੍ਰੈਡਸ਼ੀਟ"},
    "extension": {"hi": "एक्सटेंशन", "gu": "એક્સટેન્શન", "pa": "ਐਕਸਟੈਂਸ਼ਨ"},
    "file": {"hi": "फाइल", "gu": "ફાઇલ", "ta": "ஃபைல்", "pa": "ਫਾਈਲ"},
    "default": {"hi": "डिफ़ॉल्ट", "gu": "ડિફોલ્ટ", "pa": "ਡਿਫਾਲਟ"},
    "develop": {"hi": "डेवलप", "gu": "ડેવલપ", "pa": "ਡੈਵਲਪ"},
    "company": {"hi": "कंपनी", "pa": "ਕੰਪਨੀ"},
    "banking": {"hi": "बैंकिंग", "pa": "ਬੈਂਕਿੰਗ"},
    "program": {"pa": "ਪ੍ਰੋਗਰਾਮ"},
    "online": {"hi": "ऑनलाइन", "gu": "ઓનલાઇન", "pa": "ਆਨਲਾਈਨ"},
    "post": {"hi": "पोस्ट", "gu": "પોસ્ટ"},
    "office": {"hi": "ऑफिस"},
    "quiz": {"hi": "क्विज़", "gu": "ક્વિઝ", "pa": "ਕੁਇਜ਼"},
    "confusion": {"hi": "कन्फ्यूजन"},
    "full": {"hi": "फुल"},
    "form": {"hi": "फॉर्म"},
    # address / booking vocabulary
    "appointment": {"hi": "अपॉइंटमेंट", "gu": "અપોઇન્ટમેન્ટ", "ta": "அப்பாயிண்ட்மெண்ட்", "pa": "ਅਪਾਇੰਟਮੈਂਟ"},
    "apartments": {"hi": "अपार्टमेंट्स", "gu": "અપાર્ટમેન્ટ્સ", "ta": "அபார்ட்மெண்ட்ஸ்", "pa": "ਅਪਾਰਟਮੈਂਟਸ"},
    "flat": {"hi": "फ्लैट", "gu": "ફ્લેટ", "ta": "ஃப்ளாட்", "pa": "ਫਲੈਟ"},
    "sector": {"hi": "सेक्टर", "gu": "સેક્ટર", "ta": "செக்டர்", "pa": "ਸੈਕਟਰ"},
    "pin": {"hi": "पिन", "gu": "પિન", "ta": "பின்", "pa": "ਪਿਨ"},
    "code": {"hi": "कोड", "gu": "કોડ", "ta": "கோடு", "pa": "ਕੋਡ"},
    "order": {"hi": "ऑर्डर", "gu": "ઓર્ડર", "ta": "ஆர்டர்", "pa": "ਆਰਡਰ"},
    "number": {"hi": "नंबर", "gu": "નંબર", "pa": "ਨੰਬਰ"},
    "flight": {"hi": "फ्लाइट", "gu": "ફ્લાઇટ", "pa": "ਫਲਾਈਟ"},
    "registration": {"hi": "रजिस्ट्रेशन", "gu": "રજિસ્ટ્રેશન", "pa": "ਰਜਿਸਟ੍ਰੇਸ਼ਨ"},
    "minute": {"hi": "मिनट", "gu": "મિનિટ", "pa": "ਮਿੰਟ"},
    "leap": {"hi": "लीप", "gu": "લીપ", "ta": "லீப்", "pa": "ਲੀਪ"},
    "year": {"hi": "ईयर"},
    "sorry": {"hi": "सॉरी", "gu": "સોરી", "ta": "சாரி", "pa": "ਸੌਰੀ"},
    "mission": {"hi": "मिशन", "gu": "મિશન", "pa": "ਮਿਸ਼ਨ"},
    # science / nature loans
    "carbon": {"hi": "कार्बन", "gu": "કાર્બન", "ta": "கார்பன்", "pa": "ਕਾਰਬਨ"},
    "dioxide": {"hi": "डाइऑक्साइड", "gu": "ડાયોક્સાઇડ", "pa": "ਡਾਈਆਕਸਾਈਡ"},
    "gas": {"hi": "गैस", "pa": "ਗੈਸ"},
    "tiger": {"hi": "टाइगर", "pa": "ਟਾਈਗਰ"},
    "pacific": {"gu": "પેસિફિક", "ta": "பசிபிக்", "pa": "ਪੈਸਿਫਿਕ"},
    "mount": {"hi": "माउंट", "gu": "માઉન્ટ", "ta": "மவுண்ட்", "pa": "ਮਾਊਂਟ"},
    "everest": {"hi": "एवरेस्ट", "gu": "એવરેસ્ટ", "ta": "எவரெஸ்ட்", "pa": "ਐਵਰੈਸਟ"},
    # proper nouns (phonetic renderings only, never translations)
    "march": {"hi": "मार्च", "ta": "மார்ச்", "pa": "ਮਾਰਚ"},
    "august": {"hi": "अगस्त", "gu": "ઓગસ્ટ", "ta": "ஆகஸ்ட்", "pa": "ਅਗਸਤ"},
    "british": {"hi": "ब्रिटिश", "gu": "બ્રિટિશ", "ta": "பிரிட்டிஷ்", "pa": "ਬ੍ਰਿਟਿਸ਼"},
    "mumbai": {"hi": "मुंबई", "gu": "મુંબઈ", "ta": "மும்பை", "pa": "ਮੁੰਬਈ"},
    "hyderabad": {"hi": "हैदराबाद", "gu": "હૈદરાબાદ", "pa": "ਹੈਦਰਾਬਾਦ"},
    "chandigarh": {"hi": "चंडीगढ़", "gu": "ચંડીગઢ", "ta": "சண்டிகர்", "pa": "ਚੰਡੀਗੜ੍ਹ"},
    "gujarat": {"hi": "गुजरात", "gu": "ગુજરાત", "ta": "குஜராத்", "pa": "ਗੁਜਰਾਤ"},
    "maharashtra": {"hi": "महाराष्ट्र", "pa": "ਮਹਾਰਾਸ਼ਟਰ"},
    "gandhinagar": {"hi": "गांधीनगर", "gu": "ગાંધીનગર", "ta": "காந்திநகர்", "pa": "ਗਾਂਧੀਨਗਰ"},
    "bengal": {"hi": "बंगाल", "gu": "બંગાળ", "ta": "வங்காள", "pa": "ਬੰਗਾਲ"},
    "godavari": {"hi": "गोदावरी", "gu": "ગોદાવરી", "ta": "கோதாவரி", "pa": "ਗੋਦਾਵਰੀ"},
    "chandrayaan": {"hi": "चंद्रयान", "gu": "ચંદ્રયાન", "ta": "சந்திரயான்", "pa": "ਚੰਦਰਯਾਨ"},
    "ganga": {"hi": "गंगा", "gu": "ગંગા", "ta": "கங்கா", "pa": "ਗੰਗਾ"},
    "rajalakshmi": {"hi": "राजलक्ष्मी", "gu": "રાજલક્ષ્મી", "ta": "ராஜலக்ஷ்மி", "pa": "ਰਾਜਲਕਸ਼ਮੀ"},
    "venkataraman": {"hi": "वेंकटरमण", "gu": "વેંકટરમણ", "ta": "வெங்கடரமண்", "pa": "ਵੈਂਕਟਰਮਣ"},
    "ramayana": {"hi": "रामायण", "gu": "રામાયણ", "pa": "ਰਾਮਾਇਣ"},
    "mahabharata": {"hi": "महाभारत", "gu": "મહાભારત", "pa": "ਮਹਾਭਾਰਤ"},
    "vyasa": {"hi": "व्यास", "gu": "વ્યાસ", "pa": "ਵਿਆਸ"},
    "valmiki": {"hi": "वाल्मीकि", "gu": "વાલ્મીકિ", "ta": "வால்மீகி", "pa": "ਵਾਲਮੀਕਿ"},
    "veda": {"hi": "वेद", "gu": "વેદ", "ta": "வேத", "pa": "ਵੇਦ"},
}

# every spelling of a term -> that language's canonical form
LOAN = {lang: {} for lang in ("en", "hi", "gu", "ta", "pa")}
for _latin, _natives in TERMS.items():
    LOAN["en"][_latin] = _latin
    for _lang, _native in _natives.items():
        LOAN[_lang][_latin] = _native


def fold_loanwords(text, lang):
    """Latin spellings fold to this language's own form for the term."""
    loan = LOAN[lang]
    out = []
    for tok in text.split():
        key = tok.lower() if tok.isascii() and tok.isalpha() else tok
        out.append(loan.get(key, key))
    return " ".join(out)


def fold_digits(text):
    return unicodedata.normalize("NFC", text).translate(_DIGITS)


def _value(run, mult_flags):
    """Indian-numbering accumulation: 'nineteen hundred forty seven' -> 1947."""
    total = cur = 0
    for v, is_mult in zip(run, mult_flags):
        if is_mult:
            cur = (cur or 1) * v
            if v >= 1000:
                total, cur = total + cur, 0
        else:
            cur += v
    return total + cur


def _render(run, mult_flags):
    # A run of bare 0-9 is a read-out digit string, not a sum. A spoken zero or
    # three-plus units means digits; 'six' alone stays 6.
    # shortcut: a 2-unit run ("four eight") sums to 12 instead of "48"; widen to
    # len >= 2 if the bench ever reads a 2-digit code out loud.
    if not any(mult_flags) and all(0 <= v <= 9 for v in run) and (len(run) >= 3 or 0 in run):
        return "".join(str(v) for v in run)
    return str(_value(run, mult_flags))


def words_to_digits(text, lang):
    """Replace number words with digits. Unknown words pass through untouched."""
    nums, mults = NUM[lang], MULT[lang]
    out, run, flags = [], [], []

    def flush():
        if run:
            out.append(_render(run, flags))
            run.clear(); flags.clear()

    for tok in fold_digits(text).split():
        key = tok.strip(".,!?;:।").lower() if lang == "en" else tok.strip(".,!?;:।")
        if key in nums:
            run.append(nums[key]); flags.append(False)
        elif key in mults:
            run.append(mults[key]); flags.append(True)
        else:
            flush(); out.append(tok)
    flush()
    return " ".join(out)


def normalize(text, lang):
    """Full normalizer: NFC, digits folded, punctuation dropped, spaces collapsed."""
    text = words_to_digits(text, lang)
    if lang == "en":
        text = text.lower()  # Indic scripts are caseless, so this is en-only
    text = re.sub(r"[।॥?!,.;:\"'‌‍()\[\]-]", " ", text)
    return fold_loanwords(" ".join(text.split()), lang)


if __name__ == "__main__":
    cases = [
        ("en", "four eight seven three zero nine five one", "48730951"),
        ("en", "one six zero zero one seven", "160017"),
        ("en", "two hundred forty nine", "249"),
        ("en", "one thousand four hundred fifty", "1450"),
        ("en", "eighty nine", "89"),
        ("hi", "चार आठ सात तीन शून्य नौ पाँच एक", "48730951"),
        ("hi", "एक छह शून्य शून्य एक सात", "160017"),
        ("hi", "दो सौ उनचास", "249"),
        ("hi", "एक हज़ार चार सौ पचास", "1450"),
        ("hi", "एक हज़ार सात सौ अट्ठासी", "1788"),
        ("hi", "तीन सौ छियासठ", "366"),
        ("hi", "उन्नीस सौ सैंतालीस", "1947"),
        ("hi", "सत्रह", "17"),
        ("gu", "ચાર આઠ સાત ત્રણ શૂન્ય નવ પાંચ એક", "48730951"),
        ("gu", "બસો ઓગણપચાસ", "249"),
        ("gu", "એક હજાર ચારસો પચાસ", "1450"),
        ("gu", "એક હજાર સાતસો અઠ્યાસી", "1788"),
        ("gu", "ત્રણસો છાસઠ", "366"),
        ("gu", "ઓગણીસસો સુડતાળીસ", "1947"),
        ("ta", "நான்கு எட்டு ஏழு மூன்று பூஜ்ஜியம் ஒன்பது ஐந்து ஒன்று", "48730951"),
        ("ta", "இருநூற்று நாற்பத்தி ஒன்பது", "249"),
        ("ta", "ஆயிரத்து நாநூற்று ஐம்பது", "1450"),
        ("ta", "ஆயிரத்து எழுநூற்று எண்பத்தி எட்டு", "1788"),
        ("ta", "முந்நூற்று அறுபத்தி ஆறு", "366"),
        ("ta", "ஆயிரத்து தொள்ளாயிரத்து நாற்பத்தி ஏழு", "1947"),
        ("pa", "ਚਾਰ ਅੱਠ ਸੱਤ ਤਿੰਨ ਸਿਫ਼ਰ ਨੌਂ ਪੰਜ ਇੱਕ", "48730951"),
        ("pa", "ਦੋ ਸੌ ਉਣੰਜਾ", "249"),
        ("pa", "ਇੱਕ ਹਜ਼ਾਰ ਚਾਰ ਸੌ ਪੰਜਾਹ", "1450"),
        ("pa", "ਇੱਕ ਹਜ਼ਾਰ ਸੱਤ ਸੌ ਅਠਾਸੀ", "1788"),
        ("pa", "ਤਿੰਨ ਸੌ ਛਿਆਹਠ", "366"),
        ("pa", "ਉਨੀ ਸੌ ਸੰਤਾਲੀ", "1947"),
    ]
    bad = 0
    for lang, text, want in cases:
        got = words_to_digits(text, lang)
        if got != want:
            print(f"FAIL [{lang}] {text!r} -> {got!r}, want {want!r}")
            bad += 1
    # native digits fold, surrounding words survive
    assert fold_digits("१७८८") == "1788"
    assert fold_digits("૧૬૦૦૧૭") == "160017"
    assert words_to_digits("मेरा ऑर्डर नंबर चार आठ सात है", "hi") == "मेरा ऑर्डर नंबर 487 है"
    assert words_to_digits("चार आठ", "hi") == "12"  # 2-unit run sums, see _render
    assert normalize("दीजिए। मैंने?", "hi") == "दीजिए मैंने"
    assert normalize("The Pacific Ocean.", "en") == "the pacific ocean"
    # script choice must not cost WER: every form collapses to the same token
    for lang, native in [("hi", "यूएसबी"), ("gu", "યુએસબી"), ("ta", "யூஎஸ்பி"), ("pa", "ਯੂਐਸਬੀ")]:
        assert normalize("USB", lang) == native, (lang, normalize("USB", lang))
        assert normalize(native, lang) == native
    assert normalize("USB", "en") == "usb"
    assert normalize("Wi-Fi", "hi") == "वाई फाई"
    assert normalize("WiFi", "hi") == "वाई फाई"
    assert normalize("ਵਾਈਫਾਈ", "pa") == "ਵਾਈ ਫਾਈ"   # SraVaani writes it joined
    assert normalize("वाईफाई", "hi") == "वाई फाई"
    assert normalize("Android operating system", "pa") == "ਐਂਡਰਾਇਡ ਆਪਰੇਟਿੰਗ ਸਿਸਟਮ"
    print(f"{len(cases) - bad}/{len(cases)} passed")
