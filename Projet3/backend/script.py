#!/usr/bin/env python3
import os
import re
import sys
import time
import csv
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

GUTENBERG_HTML_TEMPLATE = "https://www.gutenberg.org/cache/epub/{id}/pg{id}-images.html"
DEFAULT_CSV_PATH = "bookCatalog.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Ingestion Project Gutenberg (CSV + HTML) -> Postgres")
    p.add_argument(
        "--db",
        dest="db_dsn",
        default=os.getenv("DATABASE_URL", "postgres://appuser:secret@localhost:5432/appdb"),
        help="DSN Postgres (ex: postgres://user:pass@host:5432/dbname) [env: DATABASE_URL]",
    )
    p.add_argument(
        "--target",
        type=int,
        default=int(os.getenv("TARGET_COUNT", "1664")),
        help="Nombre de livres à insérer (def: 1664)",
    )
    p.add_argument(
        "--min-words",
        type=int,
        default=int(os.getenv("MIN_WORDS", "10000")),
        help="Nombre minimal de mots par livre (def: 10000)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("DOWNLOAD_TIMEOUT", "30")),
        help="Timeout HTTP secondes (def: 30)",
    )
    p.add_argument(
        "--csv",
        dest="csv_path",
        default=os.getenv("BOOK_CATALOG_CSV", DEFAULT_CSV_PATH),
        help=f"Chemin vers le CSV du catalogue (def: {DEFAULT_CSV_PATH})",
    )
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
    DECLARE
      cfg regconfig;
      content_snippet text;
    BEGIN
      -- On mappe le champ lang vers une config de recherche Postgres
      cfg :=
        CASE lower(COALESCE(NEW.lang, ''))
          WHEN 'en' THEN 'english'::regconfig
          WHEN 'fr' THEN 'french'::regconfig
          ELSE 'simple'::regconfig
        END;

      -- On tronque le contenu pour éviter de dépasser la limite 1Mo de tsvector
      -- Ajuste 200000 si tu veux plus/moins.
      content_snippet := left(COALESCE(NEW.content, ''), 200000);

      NEW.tsv :=
        setweight(to_tsvector(cfg, COALESCE(NEW.title, '')), 'A') ||
        setweight(to_tsvector(cfg, content_snippet), 'B');

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



def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))


def html_to_plain_text(html: str) -> str:
    """
    Nettoyage très simple : on enlève les balises HTML et on compresse les espaces.
    Si tu veux mieux, tu peux ajouter BeautifulSoup plus tard.
    """
    # Supprimer les scripts/styles grossièrement
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    # Supprimer les balises
    text = re.sub(r"<[^>]+>", " ", html)
    # Décoder quelques entités de base
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">")
    # Compresser les espaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def upsert_books(conn, rows: List[tuple]):
    """
    rows: list of tuples (gutenberg_id, title, author, content, lang)
    """
    if not rows:
        return

    sql = """
    INSERT INTO books (gutenberg_id, title, author, content, lang)
    VALUES %s
    ON CONFLICT (gutenberg_id) DO UPDATE SET
      title     = EXCLUDED.title,
      author    = EXCLUDED.author,
      content   = EXCLUDED.content,
      lang      = EXCLUDED.lang,
      created_at = NOW();
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=50)
    conn.commit()


def download_book_html(session: requests.Session, book_id: int, timeout: int) -> Optional[str]:
    url = GUTENBERG_HTML_TEMPLATE.format(id=book_id)
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code != 200:
            print(f"[INGEST][WARN] HTTP {r.status_code} pour {url}, on skip.", flush=True)
            return None
        return r.text
    except requests.Timeout:
        print(f"[INGEST][WARN] Timeout en téléchargeant {url}, on skip.", flush=True)
        return None
    except requests.RequestException as e:
        print(f"[INGEST][WARN] Erreur en téléchargeant {url}: {e}", flush=True)
        return None


def iter_catalog_rows(csv_path: str):
    """
    Itère sur les lignes du CSV.
    On suppose que les colonnes sont au moins : Text#, Title, Authors, Language.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def main():
    args = parse_args()
    print(
        f"[INGEST] DB={args.db_dsn} target={args.target} "
        f"min_words={args.min_words} csv={args.csv_path}",
        flush=True,
    )

    if not os.path.exists(args.csv_path):
        print(f"[INGEST][ERROR] CSV introuvable: {args.csv_path}", file=sys.stderr)
        sys.exit(1)

    # Connexion DB
    try:
        conn = psycopg2.connect(args.db_dsn)
    except Exception as e:
        print(f"[INGEST][ERROR] Connexion Postgres échouée: {e}", file=sys.stderr)
        sys.exit(1)

    # S’assure que le schéma est prêt
    ensure_tables(conn)

    session = requests.Session()
    session.headers.update({"User-Agent": "Cajoue-Gutenberg-Ingest/2.0"})

    inserted_total = 0
    batch: List[tuple] = []
    seen_ids = set()

    try:
        for row in iter_catalog_rows(args.csv_path):
            # Récupère l'ID du livre depuis la colonne "Text#"
            raw_id = (row.get("Text#") or "").strip()
            if not raw_id:
                continue

            try:
                gid = int(raw_id)
            except ValueError:
                continue

            if gid in seen_ids:
                continue
            seen_ids.add(gid)

            title = (row.get("Title") or "").strip()
            author = (row.get("Authors") or "").strip()
            lang = (row.get("Language") or "").strip().lower() or "simple"

            # Télécharge la page HTML du livre
            html = download_book_html(session, gid, args.timeout)
            if not html:
                continue

            content = html_to_plain_text(html)
            if not content:
                continue

            if word_count(content) < args.min_words:
                # Trop court, on ne garde pas (pour garder des livres vraiment longs)
                continue

            batch.append((gid, title, author, content, lang))

            if len(batch) >= 20:
                upsert_books(conn, batch)
                inserted_total += len(batch)
                print(f"[INGEST] Inserted {inserted_total} / {args.target}", flush=True)
                batch.clear()

            if inserted_total >= args.target:
                break

        # Dernier batch
        if batch:
            upsert_books(conn, batch)
            inserted_total += len(batch)
            batch.clear()

        print(f"[INGEST] Done. Inserted total = {inserted_total}", flush=True)

        if inserted_total < args.target:
            print(
                f"[INGEST][WARN] Moins de livres que la cible: {inserted_total} < {args.target}",
                flush=True,
            )

    finally:
        conn.close()


if __name__ == "__main__":
    main()
