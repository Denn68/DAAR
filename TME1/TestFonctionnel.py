import sys
import subprocess

import RegEx as RegEx

FILE_PATH = "babylone.txt"

PATTERNS = [
    "Sargon",
    "S(a|g|r)+on", 
    "Sa.*on",
    "S(ar|ra)gon",
    "Sargon+",
    "Sargon|Assyria",
    "Akkad",
]

def run_custom(pattern: str, file_path: str):
    regEx = RegEx.RegEx(pattern)
    dfa = regEx.to_nfa().to_dfa().minimize()
    out = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if dfa.search(line):     
                out.append(line)
    return out

def find_grep_cmd():
    try:
        subprocess.run(["egrep", "-E"] + ["--version"], capture_output=True)
        return ["egrep", "-E"]
    except Exception:
        pass
    print("ERROR: ni 'egrep' ni 'grep -E' trouvés.", file=sys.stderr)
    sys.exit(3)

def run_egrep(pattern: str, file_path: str, grep_cmd):
    res = subprocess.run(
        grep_cmd + [pattern, file_path],
        capture_output=True, text=True, encoding="utf-8"
    )
    if res.returncode not in (0, 1):
        print(f"ERROR pattern={pattern!r} -> {res.stderr.strip()}", file=sys.stderr)
    return [ln.rstrip("\n") for ln in res.stdout.splitlines()]

def show_diff(title, lines, limit=5):
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
        with open(FILE_PATH, "r", encoding="utf-8", errors="ignore") as _:
            pass
    except Exception as e:
        print(f"ERROR: impossible d'ouvrir {FILE_PATH}: {e}", file=sys.stderr)
        sys.exit(2)

    grep_cmd = find_grep_cmd()

    all_ok = True

    for pat in PATTERNS:
        print(f"REGEX : {pat!r}")
        try:
            customResult = run_custom(pat, FILE_PATH)
        except Exception as e:
            print(f" ERROR {e}")
            print("-" * 60)
            all_ok = False
            continue

        egrepResult = run_egrep(pat, FILE_PATH, grep_cmd)

        setCustomResult   = set(customResult)
        setEgrepResult = set(egrepResult)

        posDiff   = setCustomResult - setEgrepResult
        negDiff = setEgrepResult - setCustomResult

        if not posDiff and not negDiff:
            print("OK✅")
        else:
            all_ok = False
            print("PAS OK❌")
            print("Différences:")
            show_diff("CUSTOM", posDiff)
            show_diff("EGREP ", negDiff)

        print("-" * 50)

    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
