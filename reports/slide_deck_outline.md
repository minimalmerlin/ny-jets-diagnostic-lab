# Foliensstruktur

## NY Jets Diagnostic Lab

Ziel: 6 bis 8 Slides
Format: deutsche Abschlusspräsentation
Basis: Streamlit-Dashboard + Report + Praesentationsskript

## Slide 1: Titel und Hook

Titel:
Warum verlieren die New York Jets immer wieder?

Untertitel:
Ein datenwissenschaftliches Projekt zu strukturellen Ursachen, Prognosemodellen und realistischen Verbesserungshebeln

Inhalt:
- Abschlussprojekt im Bereich Data Science
- Datenschnitt: NFL-Saisons 2016 bis 2025
- Ziel: Jets im Ligakontext diagnostizieren und Verbesserungspotenziale quantifizieren

Visual:
- Screenshot oder Live-Ansicht der `Executive Summary`

Sprechnotiz:
- Einstieg ueber die bekannte Jets-Erzaehlung
- Dann sofort die Umdeutung: nicht Fan-Narrativ, sondern datenwissenschaftliche Frage

## Slide 2: Forschungsfrage und Datenbasis

Titel:
Forschungsfrage und Datenpipeline

Kernaussage:
Ich untersuche, welche messbaren Faktoren die Jets systematisch von erfolgreichen NFL-Teams unterscheiden.

Inhalt:
- Forschungsfrage: Welche Offense-, Defense- und Kontinuitaetsfaktoren erklaeren die Underperformance der Jets?
- Datenquellen: `nflreadpy` / `nflverse`
- Speicherung: SQLite
- Analyseebene: Team-Week und Team-Game
- Wichtig: nur pregame Features, kein Leakage

Visual:
- Kleine Pipeline-Grafik:
  - Rohdaten
  - Feature Engineering
  - Modelltraining
  - Streamlit-Dashboard

Sprechnotiz:
- Betonen, dass das Projekt reproduzierbar und technisch sauber aufgebaut ist

## Slide 3: Die Jets im Ligavergleich

Titel:
Die Jets sind nicht nur ungluecklich, sondern strukturell schwach

Kernaussage:
Die Jets liegen ueber Jahre klar unter dem Ligamittel.

Inhalt:
- 2025 Record: `3-14`
- Siegquote 2025: `17.6%`
- Gap zum Ligamittel: `-31.2 Prozentpunkte`
- Seit 2016 im Mittel nur `4.9 Siege`
- `0` Winning Seasons seit 2016

Visual:
- Chart aus `Liga-Ueberblick`
  - Siegquote Jets vs. Liga ueber Zeit
  - optional daneben Scatterplot mit Siege vs. Punktdifferenz

Sprechnotiz:
- Das Projekt startet bewusst mit Deskription, um das Problemfeld klar abzugrenzen

## Slide 4: Modellansatz

Titel:
Vom Ligavergleich zur Prognose

Kernaussage:
Ich nutze ein interpretiertes Basismodell und ein staerkeres Hauptmodell.

Inhalt:
- Modell A: Logistische Regression
- Modell B: Gradient Boosting
- Zielvariable: Siegwahrscheinlichkeit
- Evaluationsdesign: Rolling-Origin-Backtests
- Metriken: Log Loss, Brier Score, ROC-AUC

Visual:
- einfache Gegenueberstellung der beiden Modelle
- optional kleine Methodik-Grafik mit Train auf Vergangenheit, Test auf naechste Saison

Sprechnotiz:
- Erklaeren, warum Accuracy hier nicht reicht
- Wahrscheinlichkeitsqualitaet ist wichtiger als reine Klassifikation

## Slide 5: Modellguete und Robustheit

Titel:
Das Hauptmodell generalisiert stabil

Kernaussage:
Das Gradient-Boosting-Modell ist saisonuebergreifend besser als der naive Benchmark.

Inhalt:
- Gradient Boosting:
  - Log Loss: `0.6506`
  - Brier Score: `0.2294`
  - ROC-AUC: `0.6638`
- Naiver Benchmark:
  - Log Loss: `0.6913`
  - Brier Score: `0.2490`
- Benchmark geschlagen in `8/8` Test-Saisons
- Train/Test-Gap im Log Loss nur `0.0517`

Visual:
- `Executive Summary`
  - Modellvergleich im Mittel
  - Benchmark je Testsaison
  - Generalization Gap

Sprechnotiz:
- Das ist die methodisch wichtigste Folie
- Hier klar sagen: Overfitting war anfangs ein Problem, wurde aber durch Feature-Selektion und Regularisierung deutlich reduziert

## Slide 6: Die wichtigsten Root Causes

Titel:
Was machen die Jets konkret falsch?

Kernaussage:
Es gibt drei dominante Hebel, die wiederholt mit Niederlagen zusammenhaengen.

Inhalt:
- `offense_sack_rate`
- `defense_takeaway_rate`
- `defense_explosive_play_rate_allowed`
- weitere Signale:
  - `ngs_passing_aggressiveness`
  - `offense_success_rate`

Visual:
- Balkendiagramm aus `Root Causes`

Sprechnotiz:
- Wichtig: keine Proxy-Metriken wie reine Siege oder Punktdifferenz als Ursachen verkaufen
- Fokus auf strukturelle, beeinflussbare Leistungsmetriken

## Slide 7: What-If Simulator

Titel:
Welche Verbesserungen wuerden die Jets am meisten staerken?

Kernaussage:
Das Projekt endet nicht bei Analyse, sondern bei quantifizierten Hebeln.

Inhalt:
- Verbesserung der Protection senkt Sack-Rate
- mehr Takeaways auf Defense-Seite
- weniger zugelassene Explosive Plays
- diese Hebel verschieben die modellierte Siegwahrscheinlichkeit

Visual:
- `What-If Simulator`
  - Baseline Expected Wins vs. Scenario Expected Wins
  - Delta-Chart je Spiel

Sprechnotiz:
- Hier wird der Praxisbezug sichtbar
- Gute Formulierung: "Ich simuliere keine Fantasie-Saison, sondern verändere realistische Performance-Hebel."

## Slide 8: Fazit

Titel:
Fazit

Kernaussage:
Die Jets sind nicht verflucht. Sie verlieren wegen eines wiederkehrenden, messbaren Leistungsprofils.

Inhalt:
- strukturelle Underperformance statt reines Pech
- robuste Modellierung statt nur beschreibender Analyse
- wichtigste Hebel:
  - Protection
  - defensive Ballproduktion
  - Vermeidung grosser gegnerischer Raumgewinne

Abschluss-Satz:
Das Projekt zeigt, wie sich ein sportliches Narrativ in ein belastbares Data-Science-Produkt mit Datenbank, ML-Pipeline und Streamlit-Dashboard uebersetzen laesst.

## Empfehlung fuer die Gestaltung

- Maximal 3 Kernpunkte pro Slide
- Pro Slide nur eine Hauptbotschaft
- Moeglichst eine starke Visualisierung statt viel Text
- Zahlen nur dort zeigen, wo sie die Kernaussage tragen
- Fuer die Live-Praesentation das Streamlit-Dashboard gezielt statt permanent offen nutzen

## Empfohlene Live-Reihenfolge im Dashboard

1. `Executive Summary`
2. `Liga-Ueberblick`
3. `Executive Summary` fuer Modellfolie
4. `Root Causes`
5. `What-If Simulator`

## Wenn du nur 6 Slides willst

Dann zusammenlegen:
- Slide 1 und 2 zusammen
- Slide 4 und 5 zusammen

Dann bleibt:
1. Problem + Forschungsfrage
2. Datenbasis + Pipeline
3. Jets vs. Liga
4. Modellansatz + Modellguete
5. Root Causes + Simulator
6. Fazit
