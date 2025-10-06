import sys
import subprocess

import RegEx as RegEx

filePath = "babylone.txt"

patterns = [
    "Sargon",
    "S(a|g|r)+on",
    "Sa.*on",
    "S(ar|ra)gon",
    "Sargon+",
    "Sargon|Assyria",
    "Akkad",
]

def runCustom(pattern: str, filePath: str):
    regEx = RegEx.RegEx(pattern)
    dfa = regEx.toNfa().toDfa().minimize()
    out = []
    with open(filePath, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if dfa.search(line):
                out.append(line)
    return out

def findGrepCmd():
    try:
        subprocess.run(["egrep", "-E", "--version"], capture_output=True)
        return ["egrep", "-E"]
    except Exception:
        pass
    print("ERROR: ni 'egrep' ni 'grep -E' trouvés.", file=sys.stderr)
    sys.exit(3)

def runEgrep(pattern: str, filePath: str, grepCmd):
    res = subprocess.run(
        grepCmd + [pattern, filePath],
        capture_output=True, text=True, encoding="utf-8"
    )
    if res.returncode not in (0, 1):
        print(f"ERROR pattern={pattern!r} -> {res.stderr.strip()}", file=sys.stderr)
    return [ln.rstrip("\n") for ln in res.stdout.splitlines()]

def showDiff(title, lines, limit=5):
    if not lines:
        print(f"  {title}: (rien)")
        return
    print(f"  {title}: {len(lines)} ligne(s) (max {limit} affichées)")
    for i, ln in enumerate(sorted(lines)):
        if i >= limit:
            print("    ...")
            break
        print(f"    {ln}")

def main():
    try:
        with open(filePath, "r", encoding="utf-8", errors="ignore"):
            pass
    except Exception as e:
        print(f"ERROR: impossible d'ouvrir {filePath}: {e}", file=sys.stderr)
        sys.exit(2)

    grepCmd = findGrepCmd()

    allOk = True

    for pat in patterns:
        print(f"REGEX : {pat!r}")
        try:
            customResult = runCustom(pat, filePath)
        except Exception as e:
            print(f" ERROR {e}")
            print("-" * 60)
            allOk = False
            continue

        egrepResult = runEgrep(pat, filePath, grepCmd)

        setCustomResult = set(customResult)
        setEgrepResult = set(egrepResult)

        posDiff = setCustomResult - setEgrepResult
        negDiff = setEgrepResult - setCustomResult

        if not posDiff and not negDiff:
            print("OK✅")
        else:
            allOk = False
            print("PAS OK❌")
            print("Différences:")
            showDiff("CUSTOM", posDiff)
            showDiff("EGREP ", negDiff)

        print("-" * 50)

    sys.exit(0 if allOk else 1)

if __name__ == "__main__":
    main()
