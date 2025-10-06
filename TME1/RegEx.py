import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional, FrozenSet

@dataclass(frozen=True)
class RegExTree:
    root: int
    sub: Tuple["RegExTree", ...] = ()
    charclass: Optional[Set[str]] = None

    def _label(self) -> str:
        if self.root == RegEx.CONCAT: return "·"  
        if self.root == RegEx.ETOILE: return "*"
        if self.root == RegEx.ALTERN: return "|"
        if self.root == RegEx.DOT:    return "."
        if self.root == RegEx.PLUS:   return "+"
        if self.root == RegEx.CLASS:  return "[class]" 
        return chr(self.root)

    def __str__(self) -> str:
        if not self.sub:
            return self._label()
        return f"{self._label()}(" + ",".join(str(c) for c in self.sub) + ")"

EPSILON: Optional[str] = None  

class NFA:
    def __init__(self):
        self.start: int = -1
        self.accept: int = -1
        self.transitions: Dict[int, List[Tuple[Optional[str], int]]] = {}
        self._next_id = 0

    def new_state(self) -> int:
        s = self._next_id
        self._next_id += 1
        if s not in self.transitions:
            self.transitions[s] = []
        return s

    def add_edge(self, src: int, symbol: Optional[str], dst: int) -> None:
        self.transitions.setdefault(src, []).append((symbol, dst))
        self.transitions.setdefault(dst, self.transitions.get(dst, []))  

    def __str__(self) -> str:
        lines = []
        lines.append(f"States: {sorted(self.transitions.keys())}")
        lines.append(f"Start : {self.start}")
        lines.append(f"Accept: {self.accept}")
        lines.append("Transitions:")
        for s in sorted(self.transitions.keys()):
            for sym, t in self.transitions[s]:
                lab = "ε" if sym is None else (sym if sym != "." else ".(any)")
                lines.append(f"  {s} --{lab}--> {t}")
        return "\n".join(lines)

    def to_dfa(self) -> "DFA":
        def eps_closure(states: Set[int]) -> Set[int]:
            stack = list(states)
            closure = set(states)
            while stack:
                s = stack.pop()
                for sym, nxt in self.transitions.get(s, []):
                    if sym is None and nxt not in closure:
                        closure.add(nxt)
                        stack.append(nxt)
            return closure

        def move(states: Set[int], x: str) -> Set[int]:
            out: Set[int] = set()
            for s in states:
                for sym, nxt in self.transitions.get(s, []):
                    if sym == x:
                        out.add(nxt)
            return out

        alphabet: Set[str] = set()
        for s, edges in self.transitions.items():
            for sym, _ in edges:
                if sym is not None and sym != ".":
                    alphabet.add(sym)

        def move_dot(states: Set[int]) -> Set[int]:
            out: Set[int] = set()
            for s in states:
                for sym, nxt in self.transitions.get(s, []):
                    if sym == ".":
                        out.add(nxt)
            return out

        start_set = frozenset(eps_closure({self.start}))

        dfa = DFA()
        dfa.alphabet = set(alphabet)

        subset_to_q: Dict[FrozenSet[int], int] = {start_set: 0}
        dfa.transitions[0] = {}
        dfa.start = 0
        if self.accept in start_set:
            dfa.accepts.add(0)

        worklist: List[FrozenSet[int]] = [start_set]

        while worklist:
            S = worklist.pop()
            q = subset_to_q[S]

            for a in alphabet:
                target_core = move(S, a) | move_dot(S)
                if not target_core:
                    continue
                T = frozenset(eps_closure(target_core))
                if T not in subset_to_q:
                    subset_to_q[T] = len(subset_to_q)
                    dfa.transitions[subset_to_q[T]] = {}
                    if self.accept in T:
                        dfa.accepts.add(subset_to_q[T])
                    worklist.append(T)
                dfa.transitions[q][a] = subset_to_q[T]

          

        return dfa

class DFA:
    def __init__(self):
        self.start: int = -1
        self.accepts: Set[int] = set()
        self.transitions: Dict[int, Dict[str, int]] = {}
        self.alphabet: Set[str] = set()

    def __str__(self) -> str:
        lines = []
        states = sorted(self.transitions.keys())
        lines.append(f"States: {states}")
        lines.append(f"Start : {self.start}")
        lines.append(f"Accepts: {sorted(self.accepts)}")
        lines.append(f"Alphabet: {sorted(self.alphabet)}")
        lines.append("Transitions:")
        for q in states:
            for a, qq in sorted(self.transitions[q].items()):
                lines.append(f"  {q} --{a}--> {qq}")
        return "\n".join(lines)

    def _reachable(self) -> Set[int]:
        seen: Set[int] = set()
        stack = [self.start]
        while stack:
            q = stack.pop()
            if q in seen: 
                continue
            seen.add(q)
            for a, qq in self.transitions.get(q, {}).items():
                if qq not in seen:
                    stack.append(qq)
        return seen

    def _make_total(self) -> None:
        need_sink = False
        for q, outs in self.transitions.items():
            for a in self.alphabet:
                if a not in outs:
                    need_sink = True
                    break
            if need_sink:
                break
        if not need_sink:
            return

        sink = max(self.transitions.keys()) + 1 if self.transitions else 0
        self.transitions.setdefault(sink, {})
        for a in self.alphabet:
            self.transitions[sink][a] = sink  
        for q in list(self.transitions.keys()):
            self.transitions.setdefault(q, {})
            for a in self.alphabet:
                if a not in self.transitions[q]:
                    self.transitions[q][a] = sink

    def minimize(self) -> "DFA":
        reachable = self._reachable()
        transitions = {
            q: {a: qq for a, qq in outs.items() if qq in reachable}
            for q, outs in self.transitions.items() if q in reachable
        }
        accepts = self.accepts & reachable
        start = self.start

        self.transitions = transitions
        self.accepts = accepts
        self.start = start
        self._make_total()

        states = sorted(self.transitions.keys())
        accepts = set(self.accepts)
        non_accepts = set(states) - accepts

        P: List[Set[int]] = []
        if accepts:
            P.append(set(accepts))
        if non_accepts:
            P.append(set(non_accepts))

        changed = True
        while changed:
            changed = False
            newP: List[Set[int]] = []
            for block in P:
                sig_to_subset: Dict[Tuple[int, ...], Set[int]] = {}

                class_of: Dict[int, int] = {}
                for idx, cls in enumerate(P):
                    for q in cls:
                        class_of[q] = idx

                for q in block:
                    sig: List[int] = []
                    for a in sorted(self.alphabet):
                        qq = self.transitions[q][a]
                        sig.append(class_of[qq])
                    tup = tuple(sig)
                    sig_to_subset.setdefault(tup, set()).add(q)

                parts = list(sig_to_subset.values())
                newP.extend(parts)
                if len(parts) > 1:
                    changed = True

            P = newP

        rep_to_id: Dict[int, int] = {}
        state_to_block: Dict[int, int] = {}
        for i, block in enumerate(P):
            for q in block:
                state_to_block[q] = i

        dfa = DFA()
        dfa.alphabet = set(self.alphabet)
        dfa.start = state_to_block[self.start]
        for i, block in enumerate(P):
            dfa.transitions[i] = {}
            rep = next(iter(block))
            for a in sorted(dfa.alphabet):
                dfa.transitions[i][a] = state_to_block[self.transitions[rep][a]]
            if block & self.accepts:
                dfa.accepts.add(i)

        return dfa

    def search(self, text: str):
        results = []
        alpha = self.alphabet

        n = len(text)
        for i in range(n):
            q = self.start
            accepted_end = None
            for j in range(i, n):
                ch = text[j]
                if ch not in self.transitions.get(q, {}):
                    break
                q = self.transitions[q][ch]
                if q in self.accepts:
                    accepted_end = j + 1
            if accepted_end is not None:
                results.append((i, accepted_end, text[i:accepted_end]))
        return results

class RegEx:
    CONCAT = 0xC04CA7
    ETOILE = 0xE7011E
    ALTERN = 0xA17E54
    DOT    = 0xD07
    PLUS   = 0xA11ADD
    CLASS  = 0xC1A55

    _PREC = {  
        ETOILE: 3,  
        PLUS:   3,
        CONCAT: 2,   
        ALTERN: 1,   
    }

    def __init__(self, pattern: str):
        self.pattern = pattern
        tokens = self._tokenize(pattern)
        tokens = self._insert_concats(tokens)
        postfix = self._to_postfix(tokens)
        self.tree = self._postfix_to_tree(postfix)

    def _tokenize(self, s: str) -> List[object]:  
        out: List[object] = []
        i = 0
        n = len(s)
        while i < n:
            ch = s[i]
            if ch == '.':
                out.append(self.DOT)
                i += 1
            elif ch == '*':
                out.append(self.ETOILE)
                i += 1
            elif ch == '+':
                out.append(self.PLUS)
                i += 1
            elif ch == '|':
                out.append(self.ALTERN)
                i += 1
            elif ch == '(':
                out.append(ord('('))
                i += 1
            elif ch == ')':
                out.append(ord(')'))
                i += 1
            elif ch == '[':                            
                i += 1
                if i >= n:
                    raise ValueError("Classe non fermée '['")
                chars: Set[str] = set()
                invert = False

                start_i = i
                buf: List[str] = []
                while i < n and s[i] != ']':
                    buf.append(s[i])
                    i += 1
                if i >= n or s[i] != ']':
                    raise ValueError("Classe non fermée ']' manquante")
                i += 1

                j = 0
                m = len(buf)
                while j < m:
                    c1 = buf[j]
                    if j + 2 < m and buf[j+1] == '-' and buf[j+2] != ']':
                        c2 = buf[j+2]
                        a, b = ord(c1), ord(c2)
                        if a <= b:
                            for k in range(a, b + 1):
                                chars.add(chr(k))
                        else:
                            for k in range(b, a + 1):
                                chars.add(chr(k))
                        j += 3
                    else:
                        chars.add(c1)
                        j += 1

                out.append(("CLASS", chars))
            else:
                out.append(ord(ch))
                i += 1
        return out

    def _is_literal(self, t: object) -> bool:
        if isinstance(t, tuple) and len(t) == 2 and t[0] == "CLASS":
            return True
        return t not in (self.DOT, self.ETOILE, getattr(self, "PLUS", 0xDEADBEEF),
                        self.ALTERN, ord('('), ord(')'), self.CONCAT)

    def _insert_concats(self, tokens: List[object]) -> List[object]:
        out: List[object] = []
        for i, t in enumerate(tokens):
            out.append(t)
            if i + 1 < len(tokens):
                a, b = t, tokens[i + 1]
                a_can_end   = self._is_literal(a) or a in (self.DOT, ord(')'), self.ETOILE, getattr(self, "PLUS", None))
                b_can_begin = self._is_literal(b) or b in (self.DOT, ord('('))
                if a_can_end and b_can_begin:
                    out.append(self.CONCAT)
        return out


    def _to_postfix(self, tokens: List[object]) -> List[object]:
        out: List[object] = []
        ops: List[int] = []
        for t in tokens:
            if self._is_literal(t) or t == self.DOT:
                out.append(t)
            elif t == self.ETOILE or t == getattr(self, "PLUS", None):
                out.append(t) 
            elif t in (self.CONCAT, self.ALTERN):
                while ops and ops[-1] not in (ord('('), ord(')')) and \
                    self._PREC.get(ops[-1], 0) >= self._PREC.get(t, 0):
                    out.append(ops.pop())
                ops.append(t)
            elif t == ord('('):
                ops.append(t)
            elif t == ord(')'):
                while ops and ops[-1] != ord('('):
                    out.append(ops.pop())
                if not ops: raise ValueError("Parenthèse fermante sans ouvrante")
                ops.pop()
            else:
                raise ValueError("Token inconnu")
        while ops:
            if ops[-1] in (ord('('), ord(')')):
                raise ValueError("Parenthèses non fermées")
            out.append(ops.pop())
        return out

    def _postfix_to_tree(self, postfix: List[object]) -> RegExTree:
        stack: List[RegExTree] = []
        for t in postfix:
            if isinstance(t, tuple) and t[0] == "CLASS":           
                chars: Set[str] = t[1]
                stack.append(RegExTree(self.CLASS, charclass=set(chars)))
            elif self._is_literal(t) or t == self.DOT:
                stack.append(RegExTree(t if isinstance(t, int) else ord('?')))
            elif t == self.ETOILE:
                if not stack: raise ValueError("* sans opérande")
                a = stack.pop()
                stack.append(RegExTree(self.ETOILE, (a,)))
            elif t == getattr(self, "PLUS", None):
                if not stack: raise ValueError("+ sans opérande")
                a = stack.pop()
                stack.append(RegExTree(self.PLUS, (a,)))
            elif t in (self.CONCAT, self.ALTERN):
                if len(stack) < 2: raise ValueError("Opérateur binaire sans 2 opérandes")
                b, a = stack.pop(), stack.pop()
                stack.append(RegExTree(t, (a, b)))
            else:
                raise ValueError("Postfix token inconnu")
        if len(stack) != 1:
            raise ValueError("Construction d'arbre invalide")
        return stack[0]


    def match(self, text: str) -> bool:
        memo: Dict[Tuple[int, int, Tuple[int, ...]], Set[int]] = {}
        ends = self._advance(self.tree, 0, text, memo)
        return any(pos == len(text) for pos in ends)

    def _advance(self, node: RegExTree, i: int, s: str,
                 memo: Dict[Tuple[int, int, Tuple[int, ...]], Set[int]]) -> Set[int]:
        key = (node.root, i, tuple(id(ch) for ch in node.sub))
        if key in memo: return memo[key]
        res: Set[int] = set()

        if not node.sub:
            if i < len(s):
                if node.root == self.DOT:               
                    res = {i + 1}
                else:
                    res = {i + 1} if s[i] == chr(node.root) else set()
        elif node.root == self.CONCAT:
            left, right = node.sub
            mids = self._advance(left, i, s, memo)
            out: Set[int] = set()
            for m in mids:
                out |= self._advance(right, m, s, memo)
            res = out
        elif node.root == self.ALTERN:
            left, right = node.sub
            res = self._advance(left, i, s, memo) | self._advance(right, i, s, memo)
        elif node.root == self.ETOILE:
            (child,) = node.sub
            closure: Set[int] = {i}
            frontier: List[int] = [i]
            seen: Set[int] = set([i])
            while frontier:
                cur = frontier.pop()
                for npos in self._advance(child, cur, s, memo):
                    if npos not in seen:
                        seen.add(npos)
                        closure.add(npos)
                        frontier.append(npos)
            res = closure
        else:
            raise RuntimeError("Nœud inconnu")

        memo[key] = res
        return res


    def ascii_codes(self) -> str:
        return "[" + ",".join(str(ord(c)) for c in self.pattern) + "]"

    def to_nfa(self) -> NFA:
        def build(node: RegExTree, nfa: NFA) -> Tuple[int, int]:
            if not node.sub:
                s = nfa.new_state()
                t = nfa.new_state()
                if node.root == self.DOT:
                    nfa.add_edge(s, ".", t)
                elif node.root == self.CLASS:                      
                    for ch in (node.charclass or set()):
                        nfa.add_edge(s, ch, t)
                else:
                    nfa.add_edge(s, chr(node.root), t)
                return s, t

            if node.root == self.CONCAT:
                l_start, l_accept = build(node.sub[0], nfa)
                r_start, r_accept = build(node.sub[1], nfa)
                nfa.add_edge(l_accept, EPSILON, r_start)
                return l_start, r_accept

            if node.root == self.ALTERN:
                s = nfa.new_state()
                t = nfa.new_state()
                l_start, l_accept = build(node.sub[0], nfa)
                r_start, r_accept = build(node.sub[1], nfa)
                nfa.add_edge(s, EPSILON, l_start)
                nfa.add_edge(s, EPSILON, r_start)
                nfa.add_edge(l_accept, EPSILON, t)
                nfa.add_edge(r_accept, EPSILON, t)
                return s, t

            if node.root == self.ETOILE:
                s = nfa.new_state()
                t = nfa.new_state()
                c_start, c_accept = build(node.sub[0], nfa)
                nfa.add_edge(s, EPSILON, t)            
                nfa.add_edge(s, EPSILON, c_start)      
                nfa.add_edge(c_accept, EPSILON, t)   
                nfa.add_edge(c_accept, EPSILON, c_start)  
                return s, t

            if node.root == self.PLUS:                                  
                s = nfa.new_state()
                t = nfa.new_state()
                c_start, c_accept = build(node.sub[0], nfa)
                nfa.add_edge(s, EPSILON, c_start)  
                nfa.add_edge(c_accept, EPSILON, t)  
                nfa.add_edge(c_accept, EPSILON, c_start) 
                return s, t

            raise RuntimeError("Nœud inconnu")

        nfa = NFA()
        s, t = build(self.tree, nfa)
        nfa.start, nfa.accept = s, t
        return nfa

def main(argv: List[str]) -> None:
    if len(argv) > 1:
        pattern = argv[1]
    else:
        try:
            pattern = input("  >> Please enter a regEx: ").strip()
        except EOFError:
            pattern = ""

    print(f'  >> Parsing regEx "{pattern}".')
    if len(pattern) < 1:
        print("  >> ERROR: empty regEx.", file=sys.stderr)
    else:
        try:
            rx = RegEx(pattern)
            print(f"  >> ASCII codes: {rx.ascii_codes()}.")
            print(f"  >> Tree: {rx.tree}.")

            nfa = rx.to_nfa()
            print("  >> NFA (epsilon):")
            print(nfa)

            dfa = nfa.to_dfa()
            print("  >> DFA (determinized):")
            print(dfa)

            min_dfa = dfa.minimize()
            print("  >> DFA (minimized):")
            print(min_dfa)

            if len(argv) > 2:
                file_path = argv[2]
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

                matches = min_dfa.search(text)
                print(f"  >> Found {len(matches)} match(es).")
                for k, (i, j, m) in enumerate(matches[:10], 1):
                    start_ctx = max(0, i - 30)
                    end_ctx = min(len(text), j + 30)
                    ctx = text[start_ctx:end_ctx].replace("\n", " ")
                    print(f"    {k:02d}. [{i}:{j}] {m!r}  ...{ctx}...")
        except Exception as e:
            print(f'  >> ERROR: syntax error for regEx "{pattern}": {e}', file=sys.stderr)

    print("  >> Parsing completed.")

if __name__ == "__main__":
    main(sys.argv)
