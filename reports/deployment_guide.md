# Deployment Guide

## Ziel

Die Streamlit-App soll nicht nur lokal laufen, sondern ueber ein Git-Repository deploybar sein.

## Wichtige Realitaet

### Streamlit + Vercel

Eine native Streamlit-App ist **nicht der saubere Standard-Fit fuer Vercel**.

Vercel ist stark fuer:
- Next.js
- statische Frontends
- Python-/Node-APIs

Die aktuelle App ist aber ein laufender Streamlit-Prozess. Dafuer ist **Streamlit Community Cloud**, **Render**, **Railway** oder ein eigener Docker-Host deutlich passender.

Wenn du **unbedingt Vercel** nutzen willst, gibt es zwei realistische Wege:

1. Streamlit **nicht** auf Vercel deployen, sondern:
   - Streamlit auf Streamlit Community Cloud oder Render
   - optional Landing Page oder API auf Vercel

2. Die App **umbauen**:
   - Frontend z. B. mit Next.js
   - Backend z. B. mit FastAPI
   - dann Frontend und API auf Vercel

Fuer den aktuellen Projektstand ist **GitHub -> Streamlit Community Cloud** der sauberste Weg.

## Deployment-Artefakte

Nach dem Pipeline-Lauf werden jetzt zwei App-relevante Artefakte erzeugt:

- SQLite-App-Bundle: `artifacts/app/jets_app.sqlite`
- Modell-Bundle: `artifacts/models/model_bundle.joblib`

Die grosse Arbeitsdatenbank `db/jets_nfl.sqlite` wird fuer das Deployment **nicht** benoetigt.

## Empfohlener Deployment-Workflow

### 1. Artefakte erzeugen

```bash
uv run python -m jets_project.ingest --start-season 2016 --end-season 2025
uv run python -m jets_project.features
uv run python -m jets_project.train --preferred-end-season 2025
uv run python -m jets_project.evaluate
```

Danach liegen die deploybaren App-Daten hier:

- `artifacts/app/jets_app.sqlite`
- `artifacts/models/model_bundle.joblib`

### 2. Git-Repository vorbereiten

Falls das Projekt noch kein eigenes Repo ist:

```bash
cd "/Users/merlinmechler/Library/Mobile Documents/com~apple~CloudDocs/Data Analysis/NYJets-Loosing_streak_project"
git init
```

Danach:

```bash
git add .gitignore .streamlit/config.toml README.md pyproject.toml requirements.txt uv.lock
git add app src tests reports notebooks
git add data/reference/staff_tenure.csv
git add artifacts/app/jets_app.sqlite artifacts/models/model_bundle.joblib
git commit -m "Prepare NY Jets decision support app for deployment"
```

Falls noch kein Remote gesetzt ist:

```bash
git remote add origin <DEIN_GITHUB_REPO_URL>
git push -u origin main
```

## Streamlit Community Cloud

### Empfohlene Einstellungen

- Repository: dein GitHub-Repo
- Branch: `main`
- Main file path: `app/dashboard.py`
- Python dependencies: `requirements.txt`
- Python version: in den Advanced Settings `3.11` oder `3.12` waehlen

### Optional sinnvolle Environment Variables

Die App unterstuetzt jetzt diese Umgebungsvariablen:

- `JETS_SQLITE_PATH`
- `JETS_MODEL_BUNDLE_PATH`

Wenn du die Standardpfade im Repo beibehältst, brauchst du sie nicht zwingend. Sie sind hilfreich, falls du spaeter Artefakte anders ablegen willst.

## Render / Railway

Wenn du mehr Kontrolle willst als bei Streamlit Community Cloud:

- Repo auf GitHub pushen
- neuen Web Service anlegen
- Start Command:

```bash
streamlit run app/dashboard.py --server.port $PORT --server.address 0.0.0.0
```

Auch dort solltest du die kleinen Deployment-Artefakte verwenden, nicht die grosse lokale Arbeitsdatenbank.

## Warum nicht die grosse SQLite deployen?

Die lokale Arbeitsdatenbank `db/jets_nfl.sqlite` ist fast 1 GB gross und enthaelt auch Roh- und Zwischendaten.

Fuer die App brauchst du nur:
- Team-Zusammenfassungen
- Root Causes
- Modellmetriken
- Kalibrierungs- und Governance-Tabellen
- `game_features` fuer den Simulator

Deshalb exportiert `evaluate` jetzt automatisch die schlanke Datei:

- `artifacts/app/jets_app.sqlite`

## Wenn du spaeter doch Vercel willst

Dann empfehle ich diesen Architekturpfad:

1. Das Modell und die SQLite-Logik in ein kleines FastAPI-Backend auslagern
2. Ein Frontend mit Next.js bauen
3. Frontend und API auf Vercel deployen

Das ist ein eigenes Projekt und **kein** 1:1-Deployment der aktuellen Streamlit-App.

## Fazit

Fuer den aktuellen Stand gilt:

- **Ja zu GitHub**
- **Ja zu Streamlit Community Cloud / Render / Railway**
- **Nein zu nativer Streamlit-Deploy-Strategie auf Vercel ohne Umbau**
