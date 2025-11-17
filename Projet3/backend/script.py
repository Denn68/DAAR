#!/usr/bin/env python3
import os
import re
import sys
import time
import argparse
from typing import Optional, Dict, Any, List

try:
    from dotenv import load_dotenv  # facultatif
    load_dotenv()
except Exception:
    pass

import requests
import psycopg2
from psycopg2.extras import execute_values

GUTENDEX_URL = "https://gutendex.com/books"

def parse_args():
    p = argparse.ArgumentParser(description="Ingestion Project Gutenberg -> Postgres")
    p.add_argument("--db", dest="db_dsn", default=os.getenv("DATABASE_URL", "postgres://appuser:secret@localhost:5432/appdb"),
                   help="DSN Postgres (ex: postgres://user:pass@host:5432/dbname) [env: DATABASE_URL]")
    p.add_argument("--target", type=int, default=int(os.getenv("TARGET_COUNT", "2000")),
                   help="Nombre de livres à insérer (def: 2000)")
    p.add_argument("--min-words", type=int, default=int(os.getenv("MIN_WORDS", "10000")),
                   help="Nombre minimal de mots par livre (def: 10000)")
    p.add_argument("--langs", default=os.getenv("GUTEN_LANGS", "en,fr"),
                   help="Langues Gutendex (CSV), ex: en,fr (def: en,fr)")
    p.add_argument("--timeout", type=int, default=int(os.getenv("DOWNLOAD_TIMEOUT", "30")),
                   help="Timeout HTTP secondes (def: 30)")
    return p.parse_args()

def ensure_tables(conn):
    ddl = """
    CREATE TABLE IF NOT EXISTS books (
      id            SERIAL PRIMARY KEY,
      gutenberg_id  INTEGER UNIQUE,
      title         TEXT NOT NULL,
      author        TEXT,
      content       TEXT NOT NULL,
      lang          TEXT,
      created_at    TIMESTAMP NOT NULL DEFAULT NOW(),
      tsv           tsvector
    );

    CREATE INDEX IF NOT EXISTS books_tsv_idx ON books USING GIN (tsv);

    CREATE OR REPLACE FUNCTION books_tsv_update() RETURNS trigger AS $$
    BEGIN
      NEW.tsv :=
        setweight(to_tsvector(COALESCE(NEW.lang, 'simple'), COALESCE(NEW.title, '')), 'A') ||
        setweight(to_tsvector(COALESCE(NEW.lang, 'simple'), COALESCE(NEW.content, '')), 'B');
      RETURN NEW;
    END
    $$ LANGUAGE plpgsql;

    DROP TRIGGER IF EXISTS trg_books_tsv_update ON books;
    CREATE TRIGGER trg_books_tsv_update
    BEFORE INSERT OR UPDATE ON books
    FOR EACH ROW EXECUTE FUNCTION books_tsv_update();
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()

def pick_plaintext_url(formats: Dict[str, str]) -> Optional[str]:
    # 1) text/plain; charset=utf-8
    for k, v in formats.items():
        if k.lower().startswith("text/plain") and "utf-8" in k.lower():
            return v
    # 2) tout text/plain
    for k, v in formats.items():
        if k.lower().startswith("text/plain"):
            return v
    return None

def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))

def normalize_author(authors: List[Dict[str, Any]]) -> str:
    if not authors:
        return ""
    name = (authors[0].get("name") or "").strip()
    return name

def fetch_books_generator(langs_csv: str, timeout: int):
    session = requests.Session()
    session.headers.update({"User-Agent": "Cajoue-Gutenberg-Ingest/1.0"})
    params = {"languages": langs_csv, "mime_type": "text/plain", "page": 1}
    next_url = GUTENDEX_URL
    while next_url:
        r = session.get(next_url, params=params if next_url == GUTENDEX_URL else None, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        for item in data.get("results", []):
            yield item, session, timeout
        next_url = data.get("next")

def upsert_books(conn, rows: List[tuple]):
    """
    rows: list of tuples (gutenberg_id, title, author, content, lang)
    """
    sql = """
    INSERT INTO books (gutenberg_id, title, author, content, lang)
    VALUES %s
    ON CONFLICT (gutenberg_id) DO UPDATE SET
      title   = EXCLUDED.title,
      author  = EXCLUDED.author,
      content = EXCLUDED.content,
      lang    = EXCLUDED.lang,
      created_at = NOW();
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=50)
    conn.commit()

def download_text(session: requests.Session, url: str, timeout: int) -> Optional[str]:
    # Évite les .zip
    if url.endswith(".zip"):
        return None
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        txt = r.text
        # Certains dumps contiennent l’entête/footers Gutenberg : on peut les laisser ou nettoyer plus tard
        return txt
    except requests.RequestException:
        return None

def main():
    args = parse_args()
    print(f"[INGEST] DB={args.db_dsn} target={args.target} min_words={args.min_words} langs={args.langs}", flush=True)

    # Connexion DB
    try:
        conn = psycopg2.connect(args.db_dsn)
    except Exception as e:
        print(f"[INGEST][ERROR] Connexion Postgres échouée: {e}", file=sys.stderr)
        sys.exit(1)

    # S’assure que le schéma est prêt
    ensure_tables(conn)

    inserted_total = 0
    batch: List[tuple] = []
    seen_ids = set()

    try:
        for item, session, timeout in fetch_books_generator(args.langs, args.timeout):
            gid = item.get("id")
            if gid in seen_ids:
                continue
            seen_ids.add(gid)

            title = (item.get("title") or "").strip()
            author = normalize_author(item.get("authors") or [])
            languages = item.get("languages") or []
            lang = (languages[0] if languages else "").lower().strip() or "simple"

            url = pick_plaintext_url(item.get("formats", {}))
            if not url:
                continue

            text = download_text(session, url, timeout)
            if not text:
                continue

            if word_count(text) < args.min_words:
                continue

            batch.append((gid, title, author, text, lang))

            if len(batch) >= 20:
                upsert_books(conn, batch)
                inserted_total += len(batch)
                print(f"[INGEST] Inserted {inserted_total} / {args.target}", flush=True)
                batch.clear()

            if inserted_total >= args.target:
                break

        if batch:
            upsert_books(conn, batch)
            inserted_total += len(batch)
            batch.clear()

        print(f"[INGEST] Done. Inserted total = {inserted_total}", flush=True)

    finally:
        conn.close()

if __name__ == "__main__":
    main()
