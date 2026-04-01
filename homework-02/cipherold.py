from string import ascii_lowercase, ascii_uppercase


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


def form_frequencies(text):
    alphabet = ascii_lowercase
    text = text.lower()
    letter_counts = [(letter, text.count(letter)) for letter in alphabet]
    total_letters = sum(item[1] for item in letter_counts)
    if total_letters == 0:
        return [(letter, 0.0) for letter in alphabet]
    letter_frequencies = [(letter, count / total_letters) for letter, count in letter_counts]
    return letter_frequencies


def metric(A, B):
    left = [item[0] for item in A]
    right = [item[0] for item in B]
    return sum(abs(left.index(letter) - right.index(letter)) for letter in ascii_lowercase)


def get_column(text, start, step):
    # punctuation advances key index, only letters returned for frequency analysis
    return "".join(text[i] for i in range(start, len(text), step) if text[i].lower() in ascii_lowercase)


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
            key_index += 1  # punctuation still consumed a key position
    return result


def minimize_keyword(keyword):
    for num in range(1, len(keyword)):
        if len(keyword) % num == 0:
            if keyword[:num] * (len(keyword) // num) == keyword:
                return keyword[:num]
    return keyword


# --- Load files ---

with open("computer.txt", "r") as f:
    computer_frequencies = form_frequencies(f.read())
with open("cipher.txt", "r", encoding="cp1252") as f:
    text = f.read()

sorted_computer = sorted(computer_frequencies, key=lambda x: x[1], reverse=True)
text2 = text.lower()

# --- Phase 1: find top 5 key lengths using first-column frequency metric ---
# For each length, shift column 0 by each letter and pick the best-matching shift.
# The best distance for that length is stored.

print("=== Phase 1: Finding top 5 key lengths ===")
length_results = []
for length in range(1, 30):
    best_distance = float('inf')
    best_letter = None
    for letter in ascii_lowercase:
        column = get_column(text2, 0, length)
        shift = ascii_lowercase.index(letter)
        decrypted_col = encrypt_caesar(column, -shift)
        frequencies = form_frequencies(decrypted_col)
        sorted_frequencies = sorted(frequencies, key=lambda x: x[1], reverse=True)
        distance = metric(sorted_computer, sorted_frequencies)
        if distance < best_distance:
            best_distance = distance
            best_letter = letter
    length_results.append((length, best_letter, best_distance))

length_results.sort(key=lambda x: x[2])
top_5_lengths = length_results[:5]
print("Top 5 lengths (length, first_letter, distance):")
for length, letter, distance in top_5_lengths:
    print(f"  length={length}  first_letter='{letter}'  distance={distance}")

# --- Phase 2: for each top length, find key letter per column ---
# For each column position, collect top 10 letters by distance metric,
# then build all keys from those candidates and score full decryption.

print("\n=== Phase 2: Finding key letters per column (top 10 candidates each) ===")

all_scored = []
for length, _, _ in top_5_lengths:
    print(f"\n  length={length}")

    # For each column position, rank all 26 letters by distance, keep top 10
    column_candidates = []
    for pos in range(length):
        column = get_column(text2, pos, length)
        col_scores = []
        for letter in ascii_lowercase:
            shift = ascii_lowercase.index(letter)
            decrypted_col = encrypt_caesar(column, -shift)
            frequencies = form_frequencies(decrypted_col)
            sorted_frequencies = sorted(frequencies, key=lambda x: x[1], reverse=True)
            distance = metric(sorted_computer, sorted_frequencies)
            col_scores.append((letter, distance))
        col_scores.sort(key=lambda x: x[1])
        column_candidates.append(col_scores[:10])
        print(f"    pos={pos}  top3={[l for l, _ in col_scores[:3]]}")

    # Best key = simply pick rank-0 letter from each column
    key = "".join(candidates[0][0] for candidates in column_candidates)
    plaintext = decrypt_vigenere(text, key)
    minimal_key = minimize_keyword(key)
    all_scored.append((len(minimal_key), minimal_key, plaintext))
    print(f"  key='{minimal_key}'  preview: {plaintext[:120]}")

# --- Phase 3: print all results ---

print("\n=== Phase 3: All results ===")
seen_keys = set()
for length, key, plaintext in all_scored:
    if key not in seen_keys:
        seen_keys.add(key)
        print(f"length={length}  key='{key}'")
        print(f"  {plaintext[:200]}")
        print()
