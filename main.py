from src.data import DataPipeline
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pipeline.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)

Path("logs").mkdir(parents=True, exist_ok=True)

def main():
    pipeline = DataPipeline()
    pipeline.raw()
    pipeline.preprocessed()
    pipeline.labels()

if __name__ == "__main__":
    logger.info("Starting pipeline...")
    main()
    logger.info("Pipeline completed successfully.")