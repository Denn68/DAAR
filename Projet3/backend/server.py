from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional, FrozenSet, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import time
import asyncio
from contextlib import asynccontextmanager

import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2 import OperationalError

# =======================
#   CODE REGEX TME 1
# =======================
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

    # --------- booléen, early-exit ----------
    def matches(self, text: str) -> bool:
        """
        True s'il existe AU MOINS UNE correspondance (substring) du motif dans `text`.
        """
        n = len(text)
        for i in range(n):
            q = self.start
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
                    return True
        return False

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

# =======================
#   APP + DB CONNECTION (lifespan)
# =======================
DATABASE_URL = os.getenv("DATABASE_URL", "postgres://appuser:secret@db:5432/appdb")
DB_POOL: Optional[SimpleConnectionPool] = None

async def _try_init_pool(max_wait: int = 45) -> None:
    """Essaie de créer le pool pendant max_wait secondes sans faire échouer le démarrage."""
    global DB_POOL
    deadline = time.time() + max_wait
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            DB_POOL = SimpleConnectionPool(minconn=1, maxconn=5, dsn=DATABASE_URL)
            print("[DB] Pool initialisé")
            return
        except OperationalError as e:
            last_err = e
            await asyncio.sleep(1)
    print(f"[DB] Pool non initialisé après {max_wait}s: {last_err}")

def _pool_ready() -> bool:
    return DB_POOL is not None

def get_conn():
    """Récupère une connexion, avec petit retry si le pool n'est pas prêt."""
    global DB_POOL
    for _ in range(5):
        if DB_POOL is not None:
            return DB_POOL.getconn()
        time.sleep(0.5)
    raise HTTPException(status_code=503, detail="Database not ready")

def put_conn(conn):
    if DB_POOL:
        DB_POOL.putconn(conn)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await _try_init_pool(max_wait=int(os.getenv("DB_CONNECT_MAX_WAIT", "45")))
    yield
    global DB_POOL
    if DB_POOL:
        DB_POOL.closeall()
        DB_POOL = None

app = FastAPI(title="Advanced Book Search API", version="1.0.0", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# =======================
#   MODELS
# =======================
class SearchRequest(BaseModel):
    pattern: str = Field(..., description="Expression régulière (syntaxe simplifiée)")
    textInput: str = Field(..., description="Texte d'entrée utilisateur (à vérifier dans chaque livre)")
    fields: List[str] = Field(default_factory=lambda: ["title", "content"])

class MatchSpan(BaseModel):
    start: int
    end: int
    text: str

class BookResult(BaseModel):
    book: Dict[str, Any]
    patternMatches: List[MatchSpan]
    textInputFound: bool

class SearchResponse(BaseModel):
    total_books_scanned: int
    total_books_matched: int
    results: List[BookResult]

# =======================
#   ROUTES
# =======================
@app.get("/")
def root():
    return {
        "name": "Advanced Book Search API",
        "routes": ["/health", "/db-status", "/search/advanced"],
    }

@app.get("/health")
def health():
    return {"ok": True, "time": time.time()}

@app.get("/db-status")
def db_status():
    if not _pool_ready():
        return {"db": "not-initialized"}
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            (one,) = cur.fetchone()
        return {"db": "ok", "result": one}
    finally:
        put_conn(conn)

# Route pour la recherche avancée par regex
@app.post("/search/advanced", response_model=SearchResponse)
def search_advanced(req: SearchRequest):
    # 1) Compile le motif RegEx via TON moteur
    try:
        regEx = RegEx(req.pattern)
        dfa = regEx.toNfa().toDfa().minimize()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Pattern invalide: {e}")

    # 2) Validation des champs demandés
    if not req.fields:
        req.fields = ["title", "content"]

    for f in req.fields:
        if f not in ALLOWED_BOOK_FIELDS:
            raise HTTPException(
                status_code=400,
                detail=f"Champ inconnu dans 'fields': {f}. Champs autorisés: {sorted(ALLOWED_BOOK_FIELDS)}"
            )

    # 3) Récupération des livres en base
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # On va chercher toutes les colonnes utiles
            cur.execute("""
                SELECT id, gutenberg_id, title, author, content, lang
                FROM books;
            """)
            rows = cur.fetchall()

        books: List[Dict[str, Any]] = []
        for (bid, gid, title, author, content, lang) in rows:
            books.append({
                "id": bid,
                "gutenberg_id": gid,
                "title": title or "",
                "author": author or "",
                "content": content or "",
                "lang": lang or "",
            })
    finally:
        put_conn(conn)

    # 4) Parcours des livres et application du DFA
    total_scanned = 0
    results: List[BookResult] = []

    for book in books:
        total_scanned += 1

        # Concatène uniquement les champs demandés (par défaut: title + content)
        parts = [str(book.get(f, "")) for f in req.fields]
        full_text = "\n".join(parts)

        # booléen: présence littérale du textInput dans le texte concaténé
        text_input_found = bool(req.textInput) and (req.textInput in full_text)

        # booléen: correspondance du motif via DFA (substring match)
        pattern_found = dfa.matches(full_text)

        if text_input_found or pattern_found:
            results.append(BookResult(
                book=book,
                patternMatches=[],        # tu as demandé juste le booléen, donc on laisse vide
                textInputFound=text_input_found
            ))

    return SearchResponse(
        total_books_scanned=total_scanned,
        total_books_matched=len(results),
        results=results,
    )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    uvicorn.run("server:app", host=host, port=port, reload=False)
