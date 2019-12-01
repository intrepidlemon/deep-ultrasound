import os

DIRECTORY = os.path.join(os.environ["DATA_DIR"], "liver-ultrasound")
CROPS = ["free", "fixed", "uncertain-free", "uncertain-fixed"]
DESCRIPTIONS = {
    "free": "free-0",
    "fixed": "fixed-0", 
    "uncertain-free": "c3-c4-free-0",
    "uncertain-fixed": "c3-c4-fixed-0", 
}
DATA_FOLDERS = {
    ("free"): os.path.join(DIRECTORY, "free"), 
    ("fixed"): os.path.join(DIRECTORY, "fixed"), 
    ("uncertain-free"): os.path.join(DIRECTORY, "c3-c4-free"), 
    ("uncertain-fixed"): os.path.join(DIRECTORY, "c3-c4-fixed"), 
}
MODALITIES = CROPS
MODALITY_KEY = {
    "free": "Free - Complete",
    "fixed": "Fixed - Complete", 
    "uncertain-free": "Free - Uncertain",
    "uncertain-fixed": "Fixed - Uncertain", 
}
