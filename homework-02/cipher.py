from string import ascii_lowercase, ascii_uppercase

ENGLISH_FREQUENCIES = {
    'a': 0.08167, 'b': 0.01492, 'c': 0.02782, 'd': 0.04253, 'e': 0.12702,
    'f': 0.02228, 'g': 0.02015, 'h': 0.06094, 'i': 0.06966, 'j': 0.00153,
    'k': 0.00772, 'l': 0.04025, 'm': 0.02406, 'n': 0.06749, 'o': 0.07507,
    'p': 0.01929, 'q': 0.00095, 'r': 0.05987, 's': 0.06327, 't': 0.09056,
    'u': 0.02758, 'v': 0.00978, 'w': 0.02360, 'x': 0.00150, 'y': 0.01974,
    'z': 0.00074
}

ENGLISH_BIGRAMS = {
    'th': 0.0356, 'he': 0.0307, 'in': 0.0243, 'er': 0.0205, 'an': 0.0199,
    're': 0.0185, 'on': 0.0176, 'en': 0.0175, 'at': 0.0149, 'es': 0.0145,
    'ed': 0.0145, 'it': 0.0143, 'ou': 0.0140, 'ha': 0.0130, 'to': 0.0128,
    'or': 0.0128, 'is': 0.0128, 'hi': 0.0127, 'ng': 0.0120, 'ar': 0.0113,
    'te': 0.0111, 'ti': 0.0111, 'as': 0.0110, 'nd': 0.0110, 'of': 0.0108,
    'st': 0.0105, 'nt': 0.0104, 'le': 0.0101, 'io': 0.0100, 've': 0.0099,
    'co': 0.0096, 'me': 0.0096, 'de': 0.0095, 'ro': 0.0093, 'li': 0.0092,
    'ri': 0.0092, 'al': 0.0089, 'se': 0.0089, 'si': 0.0088, 'om': 0.0088,
    'ra': 0.0088, 'ic': 0.0085, 'ne': 0.0085, 'la': 0.0083, 'il': 0.0082,
    'no': 0.0082, 'ns': 0.0082, 'be': 0.0081, 'wi': 0.0080, 'di': 0.0079,
}


def encrypt_caesar(plaintext: str, shift: int = 3) -> str:
    ciphertext = ""
    alphabet = ascii_lowercase
    alphabet_upcase = ascii_uppercase
    n = len(alphabet)
    for character in plaintext:
        if character in alphabet:
            ciphertext += alphabet[(alphabet.index(character) + shift) % n]
        elif character in alphabet_upcase:
            ciphertext += alphabet_upcase[(alphabet_upcase.index(character) + shift) % n]
        else:
            ciphertext += character
    return ciphertext


def get_column(text, start, step):
    return "".join(
        text[i] for i in range(start, len(text), step)
        if text[i].lower() in ascii_lowercase
    )


def index_of_coincidence(text):
    text = "".join(c for c in text.lower() if c in ascii_lowercase)
    n = len(text)
    if n < 2:
        return 0.0
    counts = [text.count(letter) for letter in ascii_lowercase]
    return sum(f * (f - 1) for f in counts) / (n * (n - 1))


def chi_squared(text):
    text = "".join(c for c in text.lower() if c in ascii_lowercase)
    n = len(text)
    if n == 0:
        return float('inf')
    counts = [text.count(letter) for letter in ascii_lowercase]
    return sum(
        (counts[i] - n * ENGLISH_FREQUENCIES[letter]) ** 2 / (n * ENGLISH_FREQUENCIES[letter])
        for i, letter in enumerate(ascii_lowercase)
    )


def bigram_score(text):
    text = "".join(c for c in text.lower() if c in ascii_lowercase)
    if len(text) < 2:
        return 0.0
    bigrams = [text[i:i+2] for i in range(len(text) - 1)]
    return sum(ENGLISH_BIGRAMS.get(bg, 0.0) for bg in bigrams) / len(bigrams)


def find_key_length(text, max_length=30):
    results = []
    for length in range(1, max_length + 1):
        columns = [get_column(text, pos, length) for pos in range(length)]
        avg_ioc = sum(index_of_coincidence(col) for col in columns) / length
        results.append((length, avg_ioc))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def find_key(text, length):
    key = ""
    for pos in range(length):
        column = get_column(text, pos, length)
        scores = [
            (letter, chi_squared(encrypt_caesar(column, -ascii_lowercase.index(letter))))
            for letter in ascii_lowercase
        ]
        scores.sort(key=lambda x: x[1])
        key += scores[0][0]
    return key


def decrypt_vigenere(ciphertext, key):
    result = ""
    key_len = len(key)
    key_index = 0
    for char in ciphertext:
        if char.lower() in ascii_lowercase:
            shift = ascii_lowercase.index(key[key_index % key_len])
            result += encrypt_caesar(char, -shift)
            key_index += 1
        else:
            result += char
            key_index += 1  
    return result


with open("cipher.txt", "r", encoding="cp1252") as f:
    text = f.read()

print("=== Phase 1: Key length candidates (IoC) ===")
length_results = find_key_length(text)
top_lengths = [length for length, _ in length_results[:30]]
print(f"Top 30 lengths by IoC: {top_lengths}\n")

print("=== Phase 2: Decrypt each candidate, score with bigrams ===\n")
scored = []
for length in top_lengths:
    key = find_key(text, length)
    plaintext = decrypt_vigenere(text, key)
    score = bigram_score(plaintext)
    scored.append((score, length, key, plaintext))

scored.sort(key=lambda x: x[0], reverse=True)

def minimize_keyword(keyword):
    for num in range(1, len(keyword)):
        if len(keyword) % num == 0:
            if keyword[:num] * (len(keyword) // num) == keyword:
                return keyword[:num]
    return keyword

scored.sort(key=lambda x: x[0], reverse=True)

seen_keys = set()
unique_scored = []
for score, length, key, plaintext in scored:
    minimal = minimize_keyword(key)
    if minimal not in seen_keys:
        seen_keys.add(minimal)
        unique_scored.append((score, len(minimal), minimal, plaintext))

print("Top 10 by bigram score (deduplicated):")
for score, length, key, plaintext in unique_scored[:10]:
    print(f"  length={length:2d}  key='{key}'  bigram_score={score:.5f}")
    print(f"  preview: {plaintext[:120]}")
    print()
