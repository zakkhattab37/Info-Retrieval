from ir_core.pipeline import Tokenizer

def test_arabic_tokenizer():
    tokenizer = Tokenizer()
    text = "مرحبا بكم في نظام استرجاع المعلومات. هذا اختبار للغة العربية."
    # Expected: 'مرحبا' (stemmed), 'بكم' -> 'بكم', 'نظم' (system), 'رجع' (retrieval), 'علم' (info)...
    # Stopwords 'في', 'هذا' should be removed.
    
    tokens = tokenizer.tokenize(text)
    print("Original:", text)
    print("Tokens:", [t.term for t in tokens])
    
    # Check if stopwords removed
    terms = [t.term for t in tokens]
    if 'في' in terms or 'هذا' in terms:
        print("FAIL: Stopwords not removed")
    else:
        print("PASS: Stopwords removed")

    # Check stemming (ISRI stemmer usually stems 'المعلومات' to 'علم')
    if 'علم' in terms:
        print("PASS: Stemming seems to work ('المعلومات' -> 'علم')")
    else:
        print(f"WARN: Stemming might differ, got: {terms}")

if __name__ == "__main__":
    test_arabic_tokenizer()
