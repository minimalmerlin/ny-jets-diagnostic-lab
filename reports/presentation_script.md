# Praesentationsskript

## NY Jets Diagnostic Lab

Ziel: 5 bis 7 Minuten
Format: deutsche Sprechfassung fuer Abschlussprojekt und Live-Demo im Streamlit-Dashboard

## 1. Einstieg
Zeit: 30 bis 45 Sekunden
Streamlit-Seite: `Executive Summary`

"In meinem Abschlussprojekt habe ich die Frage untersucht, ob die New York Jets einfach nur Pech haben oder ob es messbare, wiederkehrende Gruende fuer ihre dauerhafte Underperformance gibt.

Die Kernidee war, das Thema nicht als Fan-Narrativ zu behandeln, sondern als datenwissenschaftliches Problem: Welche Faktoren unterscheiden die Jets systematisch von erfolgreichen NFL-Teams, und welche Veraenderungen wuerden ihre Siegchancen am staerksten verbessern?

Dafuer habe ich ein reproduzierbares Python-Projekt mit `uv`, SQLite, Machine Learning und einem Streamlit-Dashboard aufgebaut."

## 2. Forschungsfrage und Datenbasis
Zeit: 45 bis 60 Sekunden
Streamlit-Seite: `Executive Summary`

"Die Forschungsfrage lautet: Welche messbaren Offense-, Defense- und Kontinuitaetsfaktoren erklaeren die Underperformance der Jets im Ligavergleich, und welche Hebel waeren fuer eine Trendwende am wichtigsten?

Als Datenbasis nutze ich `nflreadpy` und `nflverse` fuer die Saisons 2016 bis 2025. Die Rohdaten werden verarbeitet und in SQLite gespeichert. Darauf aufbauend habe ich Team-Week- und Game-Level-Features gebaut.

Wichtig war mir dabei, nur Informationen zu verwenden, die jeweils vor dem Spiel bekannt waren. Dadurch vermeide ich Data Leakage und kann die Modelle sauber im Backtest bewerten."

## 3. Deskriptiver Ligavergleich
Zeit: 50 bis 60 Sekunden
Streamlit-Seite: `Liga-Ueberblick`

"Der erste Befund ist: Die Jets sind nicht nur leicht unterdurchschnittlich, sondern ueber Jahre strukturell schwach.

In der Saison 2025 beenden sie die Saison mit einem Record von 3 zu 14. Das entspricht einer Siegquote von 17,6 Prozent und liegt rund 31 Prozentpunkte unter dem Ligamittel. Auch die Punktdifferenz pro Spiel ist mit minus 11,94 sehr deutlich negativ.

Noch wichtiger ist der langfristige Blick: Seit 2016 erreichen die Jets im Mittel nur 4,9 Siege pro Saison und keine einzige Winning Season. Das spricht klar gegen die These, dass es nur ein paar unglueckliche Saisons waren."

## 4. Modellansatz
Zeit: 60 Sekunden
Streamlit-Seite: `Executive Summary`

"Nach der deskriptiven Analyse habe ich zwei Modelltypen aufgebaut.

Modell A ist eine logistische Regression als interpretierbare Baseline.
Modell B ist ein Gradient-Boosting-Modell als staerkeres Hauptmodell fuer Win Probabilities.

Die Evaluation erfolgt ueber Rolling-Origin-Backtests. Das heisst: Fuer jede Testsaison wird nur auf frueheren Jahren trainiert. Bewertet werden die Modelle mit Log Loss, Brier Score und ROC-AUC.

Das ist wichtig, weil ich nicht nur wissen will, ob das Modell Teams grob richtig sortiert, sondern ob die ausgegebenen Wahrscheinlichkeiten auch kalibriert und verallgemeinerbar sind."

## 5. Modellguete und Robustheit
Zeit: 60 bis 75 Sekunden
Streamlit-Seite: `Executive Summary`

"Der wichtigste methodische Befund ist, dass das Hauptmodell nach der Ueberarbeitung jetzt robust funktioniert.

Das Gradient-Boosting-Modell erreicht einen durchschnittlichen Log Loss von 0,6506, einen Brier Score von 0,2294 und eine ROC-AUC von 0,6638. Damit ist es klar besser als der naive Heimvorteils-Benchmark, der bei 0,6913 Log Loss und 0,2490 Brier liegt.

Besonders wichtig fuer die Glaubwuerdigkeit: Das Hauptmodell schlaegt den naiven Benchmark in 8 von 8 Test-Saisons sowohl bei Log Loss als auch bei Brier.

Ich habe ausserdem den Train-Test-Gap geprueft, also einen direkten Overfitting-Check. Der mittlere Gap liegt jetzt nur noch bei 0,0517 im Log Loss. Das bedeutet: Das Modell zeigt Signal, ohne im Training unrealistisch gut und im Test instabil zu sein."

## 6. Die wichtigsten Ursachen
Zeit: 75 bis 90 Sekunden
Streamlit-Seite: `Root Causes`

"Auf Basis von Ligagap und Modellgewicht lassen sich die groessten strukturellen Probleme der Jets priorisieren.

Der wichtigste Hebel ist die Offense-Sack-Rate. Die Jets liegen hier bei 0,045, die Liga bei 0,030. Das ist nicht nur ein schlechter Wert, sondern auch ein vom Modell stark gewichteter Faktor.

Der zweite Hebel ist die Takeaway-Rate der Defense. Die Jets erzwingen zu wenige Ballverluste und verlieren dadurch Feldposition, Possessions und Swing Plays.

Der dritte Hebel sind zugelassene Explosive Plays. Die Defense gibt zu viele grosse Raumgewinne ab. Dazu kommen noch schwache Offense-Erfolgsraten und auffaellige Passing-Aggressiveness-Werte, was auf schwierige Wurfsituationen und ineffiziente Down-and-Distance-Strukturen hindeutet.

Die zentrale Aussage ist also: Das Jets-Problem ist nicht ein einzelner Quarterback oder Coach, sondern ein wiederkehrendes Leistungsprofil aus schwacher Protection, zu geringer defensiver Ballproduktion und zu vielen Big Plays gegen die eigene Defense."

## 7. Was die Jets konkret verbessern muessen
Zeit: 60 Sekunden
Streamlit-Seite: `What-If Simulator`

"Der Mehrwert des Projekts liegt nicht nur im Erklaeren, sondern auch im Quantifizieren.

Im What-If-Simulator kann ich fuer die Jets realistische Verbesserungen auf zentrale Metriken anwenden und dann die veraenderte Siegwahrscheinlichkeit pro Spiel berechnen.

Die wichtigsten praktischen Empfehlungen sind:

Erstens: Die Offense-Sack-Rate muss deutlich sinken. Das bedeutet bessere O-Line-Stabilitaet, schnellere Antworten gegen Druck und mehr on-schedule Football.

Zweitens: Die Defense muss mehr Takeaways erzeugen, also mehr Ballproduktion am Catch Point und in spaeten Rotationen.

Drittens: Explosive Plays gegen die Defense muessen runter. Das betrifft Tackling, Deep-Coverage und allgemeine Struktur im Second Level.

Wenn man diese Hebel verbessert, verschiebt sich nicht nur das optische Teamprofil, sondern messbar die erwartete Win Probability."

## 8. Schluss
Zeit: 30 bis 45 Sekunden
Streamlit-Seite: `Executive Summary`

"Mein Fazit ist: Die Jets sind nicht verflucht. Sie verlieren, weil sie ueber Jahre in genau den Metriken schlecht sind, die ligaweit am staerksten mit Siegen zusammenhaengen.

Methodisch zeigt das Projekt, dass sich eine sportliche Narrative-Frage in ein belastbares Data-Science-Projekt uebersetzen laesst: mit sauberer Datenpipeline, erklaerbaren Features, validierten Modellen und einem Streamlit-Dashboard als Entscheidungshilfe.

Wenn ich das Projekt erweitern wuerde, waere der naechste Schritt ein noch tieferer Injury- und Roster-Continuity-Layer sowie eine staerkere spielzugspezifische Analyse fuer Protection und Coverage-Strukturen."

## Kurzfassung fuer 3 Minuten

Wenn die Zeit knapp wird, kannst du auf diese Reihenfolge kuerzen:

1. Problem und Forschungsfrage
2. Datenschnitt 2016 bis 2025, SQLite, pregame Features, Rolling Backtests
3. Jets 2025: 3-14, klar unter Liga, seit 2016 im Mittel 4,9 Siege
4. Hauptmodell schlaegt naiven Benchmark in 8 von 8 Test-Saisons
5. Top-Hebel: Offense-Sack-Rate, Defense-Takeaways, zugelassene Explosive Plays
6. Fazit: kein Fluch, sondern messbares, wiederkehrendes Leistungsprofil

## Typische Rueckfragen

Frage: "Warum ist das kein Overfitting?"
Antwort: "Weil ich Rolling-Origin-Backtests nutze und den Train-Test-Gap explizit kontrolliert habe. Das Hauptmodell bleibt in allen 8 Testsaisons besser als der naive Benchmark."

Frage: "Warum nicht nur deskriptiv statt ML?"
Antwort: "Weil ich nicht nur beschreiben wollte, wo die Jets schlecht sind, sondern auch quantifizieren, welche Hebel die Siegwahrscheinlichkeit am staerksten veraendern."

Frage: "Warum Streamlit?"
Antwort: "Weil ich die Ergebnisse nicht nur in einem Notebook zeigen wollte, sondern als interaktives, reproduzierbares Analyseprodukt mit Executive Summary, Root Causes und Simulator."
