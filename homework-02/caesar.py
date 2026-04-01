import typing as tp
from string import ascii_lowercase, ascii_uppercase

def encrypt_caesar(plaintext: str, shift: int = 3) -> str:
    """
    Encrypts plaintext using a Caesar cipher.

    >>> encrypt_caesar("PYTHON")
    'SBWKRQ'
    >>> encrypt_caesar("python")
    'sbwkrq'
    >>> encrypt_caesar("Python3.6")
    'Sbwkrq3.6'
    >>> encrypt_caesar("")
    ''
    """
    ciphertext = ""
    alphabet = ascii_lowercase
    alphabet_upcase = ascii_uppercase
    n = len(alphabet)
    for character in plaintext:
        if character in alphabet:
            ciphertext += alphabet[(alphabet.index(character)+shift) % n]
        elif character in alphabet_upcase:
            ciphertext += alphabet_upcase[(alphabet_upcase.index(character)+shift) % n]
        else:
            ciphertext += character
    return ciphertext


def decrypt_caesar(ciphertext: str, shift: int = 3) -> str:
    """
    Decrypts a ciphertext using a Caesar cipher.

    >>> decrypt_caesar("SBWKRQ")
    'PYTHON'
    >>> decrypt_caesar("sbwkrq")
    'python'
    >>> decrypt_caesar("Sbwkrq3.6")
    'Python3.6'
    >>> decrypt_caesar("")
    ''
    """
    plaintext = ""
    plaintext = encrypt_caesar(ciphertext, -shift)
    return plaintext


def caesar_breaker_brute_force(ciphertext: str, dictionary: tp.Set[str]) -> int:
    """
    Brute force breaking a Caesar cipher.
    """
    best_shift = 0
    best_count = 0

    for shift in range(26):
        decrypted = decrypt_caesar(ciphertext, shift)
        words = decrypted.lower().split()
        count = sum(1 for word in words if word.strip(".,!?;:") in dictionary)
        if count > best_count:
            best_count = count
            best_shift = shift
    return best_shift
