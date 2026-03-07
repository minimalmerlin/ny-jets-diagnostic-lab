# NY Jets Abschlussprojekt

Dieses Repository implementiert ein Data-Science-Abschlussprojekt zur Frage, warum die New York Jets im Ligavergleich dauerhaft unterperformen und welche messbaren Verbesserungen ihre Siegchancen am stärksten erhoehen wuerden.

## Ziele

- NFL-Daten fuer die Saisons 2016 bis 2025 zentral sammeln
- Team-, Game- und Week-Features in SQLite und Parquet aufbauen
- Injury- und Contract-/Cap-Kontext in die Features integrieren
- Zwei Prognosemodelle trainieren: Logistic Regression und Gradient Boosting
- Jets-spezifische Root-Cause-Analyse erzeugen
- Ein Streamlit-Dashboard mit Liga-Ueberblick, Tracking-Lens und What-if-Simulator bereitstellen

## Setup

```bash
uv sync
```

## Pipeline

```bash
uv run python -m jets_project.ingest --start-season 2016 --end-season 2025
uv run python -m jets_project.features
uv run python -m jets_project.train
uv run python -m jets_project.evaluate
uv run streamlit run app/dashboard.py
```

## Struktur

- `src/jets_project`: Python-Paket mit Datenpipeline, Features, Modellen und Simulator
- `data/reference/staff_tenure.csv`: manuell gepflegte Staff-/Front-Office-Referenzdaten
- `db/jets_nfl.sqlite`: kuratierte SQLite-Datenbank
- `artifacts/app/jets_app.sqlite`: schlanke Deployment-Datenbank fuer die Streamlit-App
- `reports/final_report_outline.md`: Outline fuer den schriftlichen Abschlussbericht
- `app/dashboard.py`: Streamlit-Dashboard
- `tests`: Pytest-Suite fuer Pipeline- und Modelllogik

## Hinweise

- `nflreadpy` und `lightgbm` sind Projektabhaengigkeiten. Falls sie lokal noch fehlen, installiert `uv sync` sie in die Projektumgebung.
- Der Ingestion-Teil ist so gekapselt, dass fehlende optionale Abhaengigkeiten mit klaren Fehlermeldungen abbrechen.
- Die grosse Arbeitsdatenbank `db/jets_nfl.sqlite` und Rohdaten-Caches werden bewusst nicht versioniert.
- Fuer Deployment ist die kleine App-Datenbank `artifacts/app/jets_app.sqlite` gedacht, zusammen mit `artifacts/models/model_bundle.joblib`.
- Die Deploy-Artefakte koennen ins Git-Repo aufgenommen werden; damit ist GitHub -> Streamlit Community Cloud ohne lokalen Build der Vollpipeline moeglich.
- Eine Streamlit-Deployment-Anleitung steht in `reports/deployment_guide.md`.
