## Modellübersicht & Designentscheidungen

Dieses Projekt umfasst ein **mehrstufiges Modell-Setup** zur Analyse, Vorhersage und Empfehlung von Musik‐Inhalten
auf Track- und Artist-Ebene.  
Die Modelle wurden bewusst so gewählt, dass sie **realistische Produkt-Use-Cases** abbilden
(Discovery, Ranking, Trend-Erkennung) und gleichzeitig **methodisch sauber** (zeitbewusst, leakage-frei) bleiben.

---

## Überblick über alle Modelle

### 1. Track-Modelle (supervised)

#### (A) Success Percentile within Cohort  
- **Typ:** Regression / Relevanz-Scoring  
- **Target:** `success_pct_in_cohort` (0–100)  
- **Ziel:**  
  Vorhersage der **relativen Erfolgsposition** eines Tracks innerhalb seiner Release-Kohorte
  (z. B. gleicher Monat/Jahr).

- **Warum wichtig?**  
  Absolute Popularity ist zeitabhängig und verzerrt.  
  Dieses Target ermöglicht **faire Vergleiche über Zeiträume hinweg**.

---

#### (B) Success Residual within Cohort (Overperformance)  
- **Typ:** Regression  
- **Target:** `success_residual_in_cohort`  
- **Ziel:**  
  Erkennen von Tracks, die **besser oder schlechter als der Kohorten-Durchschnitt**
  performen.

- **Warum wichtig?**  
  Identifiziert **Hidden Gems** und echte Overperformer – unabhängig von Release-Timing.

---

#### (C) Hit Prediction  
- **Typ:** Binäre Klassifikation  
- **Target:** `y_hit`  
- **Ziel:**  
  Vorhersage, ob ein Track relativ zu seinem Release-Jahr als **Hit** gilt
  (robuste, jahresrelative Schwellenwerte).

- **Warum wichtig?**  
  Liefert ein leicht interpretierbares Signal für Hit-Wahrscheinlichkeit
  und eignet sich gut für Early-Warning-Systeme.

---

#### (E) Mood Tags  
- **Typ:** Multi-Label-Klassifikation  
- **Target:** `Y_mood`  
- **Ziel:**  
  Vorhersage von **Stimmungs- und Kontext-Tags** (z. B. „energetic“, „chill“)
  auf Basis von Audio-Features.

- **Warum wichtig?**  
  Ermöglicht Playlist-Matching, Kontext-Empfehlungen und dient als
  schwache Supervision für Similarity-Modelle.

---

## 2. Empfehlungsmodelle (Track-Ebene)

#### (D) Top-K Recommender (Learning-to-Rank)  
- **Typ:** Supervised Learning-to-Rank (z. B. XGBRanker)  
- **Label:** meist (A) `success_pct_in_cohort`  
- **Gruppierung:** Release-Kohorten (`cohort_ym`)  
- **Ziel:**  
  Erzeugung **kohorteninterner Top-K Rankings** („Trending Tracks“, Discovery-Listen).

- **Warum wichtig?**  
  Empfehlungssysteme benötigen **relative Reihenfolgen**, keine absoluten Scores.
  Learning-to-Rank optimiert explizit diese Ordnung.

---

#### (E) Track-to-Track Similarity  
- **Typ:** Unüberwachtes Lernen  
- **Modell:** Autoencoder-Embeddings + Cosine Similarity  
- **Target:** *keins*  
- **Ziel:**  
  „**Songs wie dieser**“-Empfehlungen ohne Nutzerhistorie.

- **Warum wichtig?**  
  Funktioniert im **Cold-Start**, erzeugt eine kontinuierliche „Vibe-Repräsentation“
  und ergänzt Ranking-Modelle ideal.

---

## 3. Artist-Modelle (supervised)

#### (F) Artist Trajectory – Growth  
- **Typ:** Regression  
- **Target:** `y_growth`  
- **Ziel:**  
  Prognose der **zukünftigen Popularitätsentwicklung** eines Artists
  über einen festen Zeithorizont.

---

#### (G) Artist Trajectory – Breakout  
- **Typ:** Binäre Klassifikation  
- **Target:** `y_breakout` (Top-X % Wachstum innerhalb eines Jahres)  
- **Ziel:**  
  Vorhersage, ob ein Artist kurz vor einem **Breakout** steht.

- **Warum wichtig?**  
  Relevanter als „wer ist jetzt groß?“ ist oft
  „**wer wächst als Nächstes?**“.

---

## 4. Artist-Analyse (unsupervised)

#### (H) Artist Clustering  
- **Typ:** Unüberwachtes Clustering (z. B. K-Means)  
- **Input:** Aggregierte Audio-Features + Genre-Vektoren  
- **Ziel:**  
  Gruppierung ähnlicher Artists zur:
  - Segmentierung
  - Ähnlichkeitsanalyse
  - Diversitätskontrolle in Empfehlungen

---

## Entfernte Modelle & Begründung

### ❌ Track-Popularity Regression  
- **Problem:**  
  Absolute Popularity ist stark zeitabhängig und durch einfache Proxies
  (Album- oder Artist-Popularity) trivial vorherzusagen.
- **Entscheidung:**  
  Ersetzt durch **kohortenrelative Targets** (A/B), die robuster und informativer sind.

---

### ❌ Album-Popularity Regression  
- **Problem:**  
  Album-Popularity ist häufig ein Aggregat aus Track-Popularity
  und bietet wenig zusätzlichen Erkenntnisgewinn.
- **Entscheidung:**  
  Fokus auf Track- und Artist-Ebene mit höherem Mehrwert.

---

### ❌ Explicit Prediction  
- **Problem:**  
  `explicit` ist kuratiertes Metadatum, primär lyrics-basiert und bei Release bekannt.
- **Entscheidung:**  
  Kein sinnvoller Vorhersage-Use-Case mit Audio-Features allein.

---

## Zusammenfassung

Das finale Modell-Setup kombiniert:

- **Supervised Learning** für Vorhersage, Ranking und Trajektorien
- **Unsupervised Learning** für Similarity und Segmentierung
- **Zeitbewusste Targets** zur Vermeidung von Leakage und Bias

Damit entsteht ein **realistisches, produktnahes ML-System** für
Musik-Discovery, Trend-Erkennung und Artist-Analyse.
