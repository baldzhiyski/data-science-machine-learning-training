# utils/settings.py
from typing import List,Tuple

RANDOM_SEED = 12

MAIN_ALBUM_STRATEGY = "earliest_release"

# Hit label definition
HIT_PERCENTILE = 0.80
HIT_FALLBACK_POP_THRESHOLD = 60

# Genre multi-hot size
TOP_K_GENRES = 50

# Leakage / feature controls
ALLOW_TEXT_FEATURES = True
ALLOW_LEAKY_FEATURES = False

# Mood tag definitions: (tag_name, feature_name, threshold, comparator)
MOOD_TAGS: List[Tuple[str, str, float, str]] = [
    ("energetic", "energy", 0.75, "gt"),
    ("danceable", "danceability", 0.75, "gt"),
    ("acoustic", "acousticness", 0.75, "gt"),
    ("instrumental", "instrumentalness", 0.75, "gt"),
    ("happy", "valence", 0.75, "gt"),
    ("sad", "valence", 0.25, "lt"),
    ("chill", "energy", 0.25, "lt"),
]

MOOD_LABEL_SOURCE_AUDIO = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "speechiness", "valence", "loudness", "tempo",
    "key", "mode", "time_signature",
]

# Clustering
K_CLUSTERS = 30
TSNE_SAMPLE_MAX = 4000


PAST_M = 6
FUTURE_M = 6
MIN_PAST_TRACKS = 5
BREAKOUT_Q = 0.90