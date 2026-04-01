from string import ascii_lowercase, ascii_uppercase

def encrypt_vigenere(plaintext: str, keyword: str) -> str:
    """
    Encrypts plaintext using a Vigenere cipher.

    >>> encrypt_vigenere("PYTHON", "A")
    'PYTHON'
    >>> encrypt_vigenere("python", "a")
    'python'
    >>> encrypt_vigenere("ATTACKATDAWN", "LEMON")
    'LXFOPVEFRNHR'
    """
    ciphertext = ""
    c = 0
    alphabet = ascii_lowercase
    alphabet_upcase = ascii_uppercase
    n = len(alphabet)
    for character in plaintext:
        if character in alphabet:
            ciphertext += alphabet[(alphabet.index(character) + alphabet.index(keyword.lower()[c])) % n]
        elif character in alphabet_upcase:
            ciphertext += alphabet_upcase[(alphabet_upcase.index(character) + alphabet_upcase.index(keyword.upper()[c])) % n]
        else:
            ciphertext += character
        c = (c + 1) % len(keyword)
        
    return ciphertext


def decrypt_vigenere(ciphertext: str, keyword: str) -> str:
    """
    Decrypts a ciphertext using a Vigenere cipher.

    >>> decrypt_vigenere("PYTHON", "A")
    'PYTHON'
    >>> decrypt_vigenere("python", "a")
    'python'
    >>> decrypt_vigenere("LXFOPVEFRNHR", "LEMON")
    'ATTACKATDAWN'
    """
    plaintext = ""
    c = 0
    alphabet = ascii_lowercase
    alphabet_upcase = ascii_uppercase
    n = len(alphabet)
    for character in ciphertext:
        if character in alphabet:
            plaintext += alphabet[(alphabet.index(character) - alphabet.index(keyword.lower()[c])) % n]
        elif character in alphabet_upcase:
            plaintext += alphabet_upcase[(alphabet_upcase.index(character) - alphabet_upcase.index(keyword.upper()[c])) % n]
        else:
            plaintext += character
        c = (c + 1) % len(keyword)

    return plaintext
