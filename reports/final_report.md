# NY Jets Diagnostic Lab

## 1. Executive Summary

Die NYJ beenden die Saison 2025 mit einem Record von **3-14**. Damit liegt die Siegquote bei **17.6%** und damit um **-31.2%** unter dem Ligamittel. Auch die Punktdifferenz pro Spiel ist mit **-11.94** klar negativ und liegt um **-11.67** unter dem Liga-Schnitt.

Seit 2016 erreichen die Jets im Mittel nur **4.9 Siege** pro Saison und kommen nur auf **0 Winning Seasons**. Die Underperformance ist also nicht kurzfristig, sondern strukturell.

## 2. Daten und Methodik

- Datenschnitt: 2016 bis 2025
- Datenbasis: nflreadpy / nflverse, kuratiert in SQLite
- Modellierung: Logistische Regression als Baseline, Gradient-Boosting als Hauptmodell
- Evaluation: Rolling-Origin-Backtests, pro Saison nur mit historischen Informationen

## 3. Liga-Vergleich

Die Jets bleiben ueber weite Strecken des Zeitraums unter dem Ligamittel. Der Rueckstand zeigt sich nicht nur in Siegen, sondern auch in Prozessmetriken wie Protection, Effizienz und Explosive-Play-Management. Damit ist die Grundhypothese bestaetigt: Das Problem liegt nicht nur im Zufall oder in einzelnen engen Spielen.

## 4. Modellbefunde

| Modell | Log Loss | Brier Score | ROC-AUC | ECE | Benchmark-Siege |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gradient-Boosting-Modell | 0.6476 | 0.2278 | 0.6764 | 0.0683 | 7/7 |
| Logistische Regression | 0.6761 | 0.2391 | 0.6417 | 0.0785 | 6/7 |
| Naiver Heimvorteils-Benchmark | 0.6913 | 0.2491 | 0.5000 | 0.0329 | - |
Gradient-Boosting-Modell ist das tragfaehigste Hauptmodell. Es schlaegt den naiven Heimvorteils-Benchmark in **7/7** Test-Saisons bei Log Loss und ebenso in **7/7** Saisons bei Brier.

Der Generalisierungs-Check zeigt keinen kritischen Overfitting-Befund mehr. Der mittlere Train/Test-Gap des Hauptmodells liegt bei **0.0651** im Log Loss und **0.0299** im Brier Score.

Zusaetzlich liegt die mittlere Expected Calibration Error des Hauptmodells bei **0.0683**. Die Wahrscheinlichkeiten sind damit nicht nur trennscharf, sondern auch fuer Decision Support ausreichend sauber kalibriert.

Saisonale Benchmark-Vorspruenge des Hauptmodells:

- **2018**: Log-Loss-Vorsprung 0.0443, Brier-Vorsprung 0.0212
- **2019**: Log-Loss-Vorsprung 0.0261, Brier-Vorsprung 0.0138
- **2020**: Log-Loss-Vorsprung 0.0897, Brier-Vorsprung 0.0428
- **2021**: Log-Loss-Vorsprung 0.0234, Brier-Vorsprung 0.0125
- **2022**: Log-Loss-Vorsprung 0.0425, Brier-Vorsprung 0.0200
- **2023**: Log-Loss-Vorsprung 0.0119, Brier-Vorsprung 0.0062
- **2024**: Log-Loss-Vorsprung 0.0685, Brier-Vorsprung 0.0329

## 5. Wichtigste strukturelle Ursachen

- **Durchschnittlicher Cap-Anteil pro Vertrag**: Jets 0.004 vs. Liga 0.006, Problem-Score 102.0
- **Offense-Sack-Rate**: Jets 0.045 vs. Liga 0.030, Problem-Score 44.7
- **Takeaway-Rate der Defense**: Jets 0.002 vs. Liga 0.015, Problem-Score 24.7
- **Vertrags-Slots im aktiven Kader**: Jets 86.000 vs. Liga 82.844, Problem-Score 23.0
- **Offense-Erfolgsrate**: Jets 0.453 vs. Liga 0.468, Problem-Score 18.0

## 6. Handlungsempfehlungen

- Durchschnittlicher Cap-Anteil pro Vertrag gezielt in Richtung Liga-Topquartil bewegen.
- Offense-Sack-Rate gezielt in Richtung Liga-Topquartil bewegen.
- Takeaway-Rate der Defense gezielt in Richtung Liga-Topquartil bewegen.

## 7. Fazit

Die Jets verlieren nicht wegen eines einzelnen Akteurs, sondern wegen eines wiederkehrenden Profils aus Protection-Problemen, fehlender Offense-Effizienz und zu vielen zugelassenen Big Plays. Der Modellteil ist jetzt stabil genug, um diese Hebel nicht nur deskriptiv, sondern auch prognostisch zu bewerten.
