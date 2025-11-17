-- Table livres + indexation
CREATE TABLE IF NOT EXISTS books (
  id            SERIAL PRIMARY KEY,
  gutenberg_id  INTEGER UNIQUE,
  title         TEXT NOT NULL,
  author        TEXT,
  content       TEXT NOT NULL,
  lang          TEXT,                     -- ex: 'en', 'fr' (si dispo)
  created_at    TIMESTAMP NOT NULL DEFAULT NOW(),
  tsv           tsvector                  -- vecteur plein-texte
);

-- Index GIN pour la recherche plein-texte
CREATE INDEX IF NOT EXISTS books_tsv_idx ON books USING GIN (tsv);

-- Trigger pour maintenir tsv à jour (pondère le titre plus fort que le texte)
CREATE OR REPLACE FUNCTION books_tsv_update() RETURNS trigger AS $$
BEGIN
  -- Choix de la configuration de langue si connu, sinon 'simple'
  -- NB: to_tsvector(config, text)
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
