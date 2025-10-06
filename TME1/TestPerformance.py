import sys
import time
import subprocess

import RegEx as RegEx

FILE_PATH = "babylone.txt"

PATTERNS = [
    "Sargon",
    "Sa.*on",
    "S(a|g|r)+on",
    "S(ar|ra)gon",
    "Akkad",
    "(Sa|Ba)r*gon",
    "(Sargon|Assyria)",
]

def grep_cmd():
    for cmd in (["egrep", "-E"], ["grep", "-E"]):
        try:
            subprocess.run(cmd + ["--version"], capture_output=True)
            return cmd
        except Exception:
            pass
    return None

def grep_time(cmd, pattern, file_path):
    t0 = time.perf_counter()
    subprocess.run(cmd + [pattern, file_path], capture_output=True, text=True, encoding="utf-8")
    return (time.perf_counter() - t0) * 1000.0

def search_time(dfa, file_path):
    t0 = time.perf_counter()
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            dfa.search(line)
    return (time.perf_counter() - t0) * 1000.0

def main():
    try:
        with open(FILE_PATH, "r", encoding="utf-8", errors="ignore") as _:
            pass
    except Exception as e:
        print(f"ERROR opening {FILE_PATH}: {e}", file=sys.stderr)
        sys.exit(2)

    gcmd = grep_cmd()

    total_custom = 0.0
    total_grep = 0.0

    print("regex parse_ms nfa_ms dfa_ms min_ms search_ms custom_total_ms grep_ms")
    for pat in PATTERNS:
        t0 = time.perf_counter()
        regEx = RegEx.RegEx(pat)
        parse_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        nfa = regEx.to_nfa()
        nfa_ms = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        dfa = nfa.to_dfa()
        dfa_ms = (time.perf_counter() - t2) * 1000.0

        t3 = time.perf_counter()
        mindfa = dfa.minimize()
        min_ms = (time.perf_counter() - t3) * 1000.0

        search_ms = search_time(mindfa, FILE_PATH)

        custom_total = parse_ms + nfa_ms + dfa_ms + min_ms + search_ms
        total_custom += custom_total

        if gcmd is not None:
            g_ms = grep_time(gcmd, pat, FILE_PATH)
            total_grep += g_ms
            g_ms_str = f"{g_ms:.2f}"
        else:
            g_ms_str = "n/a"

        print(f"{pat} {parse_ms:.2f} {nfa_ms:.2f} {dfa_ms:.2f} {min_ms:.2f} {search_ms:.2f} {custom_total:.2f} {g_ms_str}")

    print(f"TOTAL_CUSTOM_MS\t{total_custom:.2f}")
    if gcmd is not None:
        print(f"TOTAL_GREP_MS\t{total_grep:.2f}")
    else:
        print("TOTAL_GREP_MS\tn/a")

if __name__ == "__main__":
    main()
