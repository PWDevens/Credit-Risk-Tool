import kagglehub
from dotenv import load_dotenv
from pathlib import Path
import shutil

# Load KAGGLE_USERNAME / KAGGLE_API_TOKEN from .env file
load_dotenv(Path(__file__).parent / "secrets" / ".env")

DATASET = "henryokam/prosper-loan-data"
DATA_DIR = Path(__file__).parent

# Download to kagglehub's cache and get the path to the dataset files
cache_path = Path(kagglehub.dataset_download(DATASET))

# Copy the files flat into data/ so they're easy to load 
for src in cache_path.glob("*"):
    dst = DATA_DIR / src.name
    if src.is_file():
        shutil.copy2(src, dst)
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True)


print("Dataset files copied to:", DATA_DIR.resolve())
print("Contents:", [p.name for p in DATA_DIR.glob("*") if p.name != "loader.py"])
