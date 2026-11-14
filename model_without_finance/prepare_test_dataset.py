from pathlib import Path

from db.database import SessionLocal
from db.repositories.user_repository import UserRepository, TrainingDatasetType


def prepare_dataset():
    session = SessionLocal()

    try:
        df = UserRepository().get_test_dataset_without_finances(
            dataset_type=TrainingDatasetType.PERCENT_80_RATIO_1_TO_40_AUTOBALANCED,
        )

        if df.empty:
            print("No data found.")
            return

        # Create output directory
        output_dir = Path("data")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save dataset
        output_path = output_dir / "model_test_dataset.parquet"

        df.to_parquet(output_path, index=False)

        print(f"Dataset saved: {output_path}")
        print(f"Total rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")

        print("\nTarget distribution:")
        print(df["is_vip"].value_counts())

        print("\nMissing values:")
        print(df.isnull().sum())

    finally:
        session.close()


if __name__ == "__main__":
    prepare_dataset()
