"""Column names for multi-label targets (match Data/train.csv)."""

ID_COL = "SeriesInstanceUID"

# Thirteen vessel / site columns in CSV order (before Aneurysm Present).
VESSEL_COLUMNS: tuple[str, ...] = (
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
)

# Full model / submission target columns: 15 sites + aggregate presence.
PRESENCE_COL = "Aneurysm Present"
LABEL_COLUMNS: tuple[str, ...] = VESSEL_COLUMNS + (PRESENCE_COL,)

NUM_LABELS = len(LABEL_COLUMNS)
