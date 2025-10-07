import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional, FrozenSet

@dataclass(frozen=True)
class RegExTree:
    root: int
    sub: Tuple["RegExTree", ...] = ()

EPSILON: Optional[str] = None

class NFA:
    def __init__(self):
        self.start: int = -1
        self.accept: int = -1
        self.transitions: Dict[int, List[Tuple[Optional[str], int]]] = {}
        self._nextId = 0

    def newState(self) -> int:
        s = self._nextId
        self._nextId += 1
        if s not in self.transitions:
            self.transitions[s] = []
        return s

    def addEdge(self, src: int, symbol: Optional[str], dst: int) -> None:
        self.transitions.setdefault(src, []).append((symbol, dst))
        self.transitions.setdefault(dst, self.transitions.get(dst, []))

    def toDfa(self) -> "DFA":
        def epsClosure(states: Set[int]) -> Set[int]:
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

        def moveDot(states: Set[int]) -> Set[int]:
            out: Set[int] = set()
            for s in states:
                for sym, nxt in self.transitions.get(s, []):
                    if sym == ".":
                        out.add(nxt)
            return out

        literals: Set[str] = set()
        hasDot = False
        for _, edges in self.transitions.items():
            for sym, _ in edges:
                if sym is None:
                    continue
                if sym == ".":
                    hasDot = True
                else:
                    literals.add(sym)

        startSet = frozenset(epsClosure({self.start}))

        dfa = DFA()
        dfa.alphabet = set(literals)

        subsetToQ: Dict[FrozenSet[int], int] = {startSet: 0}
        dfa.transitions[0] = {}
        dfa.start = 0
        if self.accept in startSet:
            dfa.accepts.add(0)

        worklist: List[FrozenSet[int]] = [startSet]

        while worklist:
            S = worklist.pop()
            q = subsetToQ[S]

            tdotCore = moveDot(S) if hasDot else set()
            if tdotCore:
                tdot = frozenset(epsClosure(tdotCore))
                if tdot not in subsetToQ:
                    subsetToQ[tdot] = len(subsetToQ)
                    dfa.transitions[subsetToQ[tdot]] = {}
                    if self.accept in tdot:
                        dfa.accepts.add(subsetToQ[tdot])
                    worklist.append(tdot)
                dfa.transitions[q]["."] = subsetToQ[tdot]

            for a in literals:
                targetCore = move(S, a)
                if hasDot:
                    targetCore |= tdotCore
                if not targetCore:
                    continue
                T = frozenset(epsClosure(targetCore))
                if T not in subsetToQ:
                    subsetToQ[T] = len(subsetToQ)
                    dfa.transitions[subsetToQ[T]] = {}
                    if self.accept in T:
                        dfa.accepts.add(subsetToQ[T])
                    worklist.append(T)
                dfa.transitions[q][a] = subsetToQ[T]

        return dfa

class DFA:
    def __init__(self):
        self.start: int = -1
        self.accepts: Set[int] = set()
        self.transitions: Dict[int, Dict[str, int]] = {}
        self.alphabet: Set[str] = set()

    def reachable(self) -> Set[int]:
        seen: Set[int] = set()
        stack = [self.start]
        while stack:
            q = stack.pop()
            if q in seen:
                continue
            seen.add(q)
            for _, qq in self.transitions.get(q, {}).items():
                if qq not in seen:
                    stack.append(qq)
        return seen

    def makeTotal(self) -> None:
        needSink = False
        for _, outs in self.transitions.items():
            for a in self.alphabet:
                if a not in outs:
                    needSink = True
                    break
            if "." not in outs:
                needSink = True
            if needSink:
                break
        if not needSink:
            return

        sink = max(self.transitions.keys()) + 1 if self.transitions else 0
        self.transitions.setdefault(sink, {})
        for a in self.alphabet:
            self.transitions[sink][a] = sink
        self.transitions[sink]["."] = sink
        for q in list(self.transitions.keys()):
            self.transitions.setdefault(q, {})
            for a in self.alphabet:
                if a not in self.transitions[q]:
                    self.transitions[q][a] = sink
            if "." not in self.transitions[q]:
                self.transitions[q]["."] = sink

    def minimize(self) -> "DFA":
        reachable = self.reachable()
        transitions = {
            q: {a: qq for a, qq in outs.items() if qq in reachable}
            for q, outs in self.transitions.items() if q in reachable
        }
        accepts = self.accepts & reachable
        start = self.start

        self.transitions = transitions
        self.accepts = accepts
        self.start = start
        self.makeTotal()

        states = sorted(self.transitions.keys())
        accepts = set(self.accepts)
        nonAccepts = set(states) - accepts

        P: List[Set[int]] = []
        if accepts:
            P.append(set(accepts))
        if nonAccepts:
            P.append(set(nonAccepts))

        changed = True
        while changed:
            changed = False
            newP: List[Set[int]] = []
            for block in P:
                sigToSubset: Dict[Tuple[int, ...], Set[int]] = {}

                classOf: Dict[int, int] = {}
                for idx, cls in enumerate(P):
                    for q in cls:
                        classOf[q] = idx

                for q in block:
                    sig: List[int] = []
                    for a in sorted(self.alphabet) + ["."]:
                        qq = self.transitions[q][a]
                        sig.append(classOf[qq])
                    tup = tuple(sig)
                    sigToSubset.setdefault(tup, set()).add(q)

                parts = list(sigToSubset.values())
                newP.extend(parts)
                if len(parts) > 1:
                    changed = True

            P = newP

        stateToBlock: Dict[int, int] = {}
        for i, block in enumerate(P):
            for q in block:
                stateToBlock[q] = i

        dfa = DFA()
        dfa.alphabet = set(self.alphabet)
        dfa.start = stateToBlock[self.start]
        for i, block in enumerate(P):
            dfa.transitions[i] = {}
            rep = next(iter(block))
            for a in sorted(dfa.alphabet):
                dfa.transitions[i][a] = stateToBlock[self.transitions[rep][a]]
            dfa.transitions[i]["."] = stateToBlock[self.transitions[rep]["."]]
            if block & self.accepts:
                dfa.accepts.add(i)

        return dfa

    def search(self, text: str):
        results = []
        n = len(text)
        for i in range(n):
            q = self.start
            acceptedEnd = None
            for j in range(i, n):
                ch = text[j]
                outs = self.transitions.get(q, {})
                qq = outs.get(ch)
                if qq is None:
                    qq = outs.get(".")
                if qq is None:
                    break
                q = qq
                if q in self.accepts:
                    acceptedEnd = j + 1
            if acceptedEnd is not None:
                results.append((i, acceptedEnd, text[i:acceptedEnd]))
        return results

class RegEx:
    CONCAT = 0xC04CA7
    ETOILE = 0xE7011E
    ALTERN = 0xA17E54
    DOT    = 0xD07
    PLUS   = 0xA11ADD

    _PREC = {
        ETOILE: 3,
        PLUS:   3,
        CONCAT: 2,
        ALTERN: 1,
    }

    def __init__(self, pattern: str):
        self.pattern = pattern
        tokens = self.parse(pattern)             
        tokens = self.insertConcats(tokens)
        postfix = self.toPostfix(tokens)
        self.tree = self.postfixToTree(postfix)

    def parse(self, s: str) -> List[object]:   
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
            else:
                out.append(ord(ch))
                i += 1
        return out

    def isLiteral(self, t: object) -> bool:
        return t not in (self.DOT, self.ETOILE, getattr(self, "PLUS", 0xDEADBEEF),
                         self.ALTERN, ord('('), ord(')'), self.CONCAT)

    def insertConcats(self, tokens: List[object]) -> List[object]:
        out: List[object] = []
        for i, t in enumerate(tokens):
            out.append(t)
            if i + 1 < len(tokens):
                a, b = t, tokens[i + 1]
                aCanEnd   = self.isLiteral(a) or a in (self.DOT, ord(')'), self.ETOILE, getattr(self, "PLUS", None))
                bCanBegin = self.isLiteral(b) or b in (self.DOT, ord('('))
                if aCanEnd and bCanBegin:
                    out.append(self.CONCAT)
        return out

    def toPostfix(self, tokens: List[object]) -> List[object]:
        out: List[object] = []
        ops: List[int] = []
        for t in tokens:
            if self.isLiteral(t) or t == self.DOT:
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
                if not ops:
                    raise ValueError("Parenthèse fermante sans ouvrante")
                ops.pop()
            else:
                raise ValueError("Token inconnu")
        while ops:
            if ops[-1] in (ord('('), ord(')')):
                raise ValueError("Parenthèses non fermées")
            out.append(ops.pop())
        return out

    def postfixToTree(self, postfix: List[object]) -> RegExTree:
        stack: List[RegExTree] = []
        for t in postfix:
            if self.isLiteral(t) or t == self.DOT:
                stack.append(RegExTree(t if isinstance(t, int) else ord('?')))
            elif t == self.ETOILE:
                if not stack:
                    raise ValueError("* sans opérande")
                a = stack.pop()
                stack.append(RegExTree(self.ETOILE, (a,)))
            elif t == getattr(self, "PLUS", None):
                if not stack:
                    raise ValueError("+ sans opérande")
                a = stack.pop()
                stack.append(RegExTree(self.PLUS, (a,)))
            elif t in (self.CONCAT, self.ALTERN):
                if len(stack) < 2:
                    raise ValueError("Opérateur binaire sans 2 opérandes")
                b, a = stack.pop(), stack.pop()
                stack.append(RegExTree(t, (a, b)))
            else:
                raise ValueError("Postfix token inconnu")
        if len(stack) != 1:
            raise ValueError("Construction d'arbre invalide")
        return stack[0]

    def toNfa(self) -> NFA:
        def build(node: RegExTree, nfa: NFA) -> Tuple[int, int]:
            if not node.sub:
                s = nfa.newState()
                t = nfa.newState()
                if node.root == self.DOT:
                    nfa.addEdge(s, ".", t)
                else:
                    nfa.addEdge(s, chr(node.root), t)
                return s, t

            if node.root == self.CONCAT:
                lStart, lAccept = build(node.sub[0], nfa)
                rStart, rAccept = build(node.sub[1], nfa)
                nfa.addEdge(lAccept, EPSILON, rStart)
                return lStart, rAccept

            if node.root == self.ALTERN:
                s = nfa.newState()
                t = nfa.newState()
                lStart, lAccept = build(node.sub[0], nfa)
                rStart, rAccept = build(node.sub[1], nfa)
                nfa.addEdge(s, EPSILON, lStart)
                nfa.addEdge(s, EPSILON, rStart)
                nfa.addEdge(lAccept, EPSILON, t)
                nfa.addEdge(rAccept, EPSILON, t)
                return s, t

            if node.root == self.ETOILE:
                s = nfa.newState()
                t = nfa.newState()
                cStart, cAccept = build(node.sub[0], nfa)
                nfa.addEdge(s, EPSILON, t)
                nfa.addEdge(s, EPSILON, cStart)
                nfa.addEdge(cAccept, EPSILON, t)
                nfa.addEdge(cAccept, EPSILON, cStart)
                return s, t

            if node.root == self.PLUS:
                s = nfa.newState()
                t = nfa.newState()
                cStart, cAccept = build(node.sub[0], nfa)
                nfa.addEdge(s, EPSILON, cStart)
                nfa.addEdge(cAccept, EPSILON, t)
                nfa.addEdge(cAccept, EPSILON, cStart)
                return s, t

            raise RuntimeError("Nœud inconnu")

        nfa = NFA()
        s, t = build(self.tree, nfa)
        nfa.start, nfa.accept = s, t
        return nfa


def main(argv: List[str]) -> None:
    if len(argv) < 3:
        print("usage: python RegEx.py <regex> <file>", file=sys.stderr)
        sys.exit(2)

    pattern = argv[1]
    filePath = argv[2]

    try:
        regEx = RegEx(pattern)
        minDfa = regEx.toNfa().toDfa().minimize()

        foundAny = False
        with open(filePath, "r", encoding="utf-8", errors="ignore") as f:
            for rawLine in f:
                line = rawLine.rstrip("\n")
                if minDfa.search(line):
                    print(rawLine, end="")
                    foundAny = True

        if not foundAny:
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main(sys.argv)
