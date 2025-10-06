import sys
import time
import subprocess

import RegEx as RegEx

filePath = "babylone.txt"

patterns = [
    "Sargon",
    "Sa.*on",
    "S(a|g|r)+on",
    "S(ar|ra)gon",
    "Akkad",
    "(Sa|Ba)r*gon",
    "(Sargon|Assyria)",
]

def grepCmd():
    for cmd in (["egrep", "-E"], ["grep", "-E"]):
        try:
            subprocess.run(cmd + ["--version"], capture_output=True)
            return cmd
        except Exception:
            pass
    return None

def grepTime(cmd, pattern, filePath):
    t0 = time.perf_counter()
    subprocess.run(cmd + [pattern, filePath], capture_output=True, text=True, encoding="utf-8")
    return (time.perf_counter() - t0) * 1000.0

def searchTime(dfa, filePath):
    t0 = time.perf_counter()
    with open(filePath, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            dfa.search(line)
    return (time.perf_counter() - t0) * 1000.0

def main():
    try:
        with open(filePath, "r", encoding="utf-8", errors="ignore"):
            pass
    except Exception as e:
        print(f"ERROR opening {filePath}: {e}", file=sys.stderr)
        sys.exit(2)

    gcmd = grepCmd()

    totalCustom = 0.0
    totalGrep = 0.0

    print("regex parse_ms nfa_ms dfa_ms min_ms search_ms custom_total_ms grep_ms")
    for pat in patterns:
        t0 = time.perf_counter()
        regEx = RegEx.RegEx(pat)
        parseMs = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        nfa = regEx.toNfa()
        nfaMs = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        dfa = nfa.toDfa()
        dfaMs = (time.perf_counter() - t2) * 1000.0

        t3 = time.perf_counter()
        minDfa = dfa.minimize()
        minMs = (time.perf_counter() - t3) * 1000.0

        searchMs = searchTime(minDfa, filePath)

        customTotal = parseMs + nfaMs + dfaMs + minMs + searchMs
        totalCustom += customTotal

        if gcmd is not None:
            gMs = grepTime(gcmd, pat, filePath)
            totalGrep += gMs
            gMsStr = f"{gMs:.2f}"
        else:
            gMsStr = "n/a"

        print(f"{pat} {parseMs:.2f} {nfaMs:.2f} {dfaMs:.2f} {minMs:.2f} {searchMs:.2f} {customTotal:.2f} {gMsStr}")

    print(f"TOTAL_CUSTOM_MS\t{totalCustom:.2f}")
    if gcmd is not None:
        print(f"TOTAL_GREP_MS\t{totalGrep:.2f}")
    else:
        print("TOTAL_GREP_MS\tn/a")

if __name__ == "__main__":
    main()
