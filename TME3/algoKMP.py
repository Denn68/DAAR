# kmp.py

from typing import List

def retenue(pattern: str) -> List[int]:
    """
    Calcule la liste 'Retenue' (a.k.a. LPS: Longest Proper Prefix which is also Suffix)
    pour chaque position du motif.
    """
    lps = [0] * len(pattern)
    j = 0  # longueur du plus long préfixe-suffixe courant

    # i démarre à 1 (le lps[0] est forcément 0)
    for i in range(1, len(pattern)):
        while j > 0 and pattern[i] != pattern[j]:
            j = lps[j - 1]           # on recule avec la retenue précédente
        if pattern[i] == pattern[j]:
            j += 1
            lps[i] = j
    return lps


def kmp_search(text: str, pattern: str) -> List[int]:
    """
    Retourne la liste des indices de début d’occurrence de 'pattern' dans 'text'
    en utilisant la liste 'Retenue' (LPS).
    """
    if not pattern:
        return list(range(len(text) + 1))  # motif vide: correspond entre chaque char

    lps = retenue(pattern)
    res = []
    j = 0  # index dans pattern

    for i in range(len(text)):  # i parcourt le texte
        while j > 0 and text[i] != pattern[j]:
            j = lps[j - 1]
        if text[i] == pattern[j]:
            j += 1
            if j == len(pattern):
                res.append(i - j + 1)
                j = lps[j - 1]  # on continue pour trouver d'autres occurrences
    return res


def highlight(text: str, starts: List[int], m: int) -> str:
    """
    Retourne une version du texte avec les occurrences encadrées par [ ].
    """
    if not starts:
        return text
    parts = []
    last = 0
    for s in starts:
        parts.append(text[last:s])
        parts.append("[" + text[s:s+m] + "]")
        last = s + m
    parts.append(text[last:])
    return "".join(parts)


def pretty_print_retenue(pattern: str, lps: List[int]) -> None:
    print("Motif :", repr(pattern))
    print("Index :", " ".join(f"{i:>2}" for i in range(len(pattern))))
    print("Chars :", " ".join(f"{c:>2}" for c in pattern))
    print("LPS   :", " ".join(f"{v:>2}" for v in lps))
    print()


if __name__ == "__main__":
    # Démo rapide avec les exemples suggérés
    text = "Pony Tracks"
    tests = ["Chihuahua", "Pizzi", "Pepperoni", "Pizza", "Poppers", "Pony", "rack"]

    print("Texte :", repr(text))
    print()

    for pat in tests:
        lps = retenue(pat)
        occ = kmp_search(text, pat)
        pretty_print_retenue(pat, lps)
        if occ:
            print(f"→ {pat!r} trouvé aux positions {occ}")
            print("   ", highlight(text, occ, len(pat)))
        else:
            print(f"→ {pat!r} introuvable dans le texte.")
        print("-" * 60)