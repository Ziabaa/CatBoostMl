from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db.database import SessionLocal

SQL_QUERY = """
            SELECT
                -- Financial features
                ff.avg_bet_size_per_spin,
                ff.spins_count_ftd_day,
                ff.failed_deposits_count,
                ff.deposits_sum_ftd_day,
                ff.transactions_count_ftd_day,
                ff.avg_transaction_check,

                -- Categorical feature
                m.email_domain,

                -- Time features
                tf.ftd_day_of_month,
                tf.ftd_month_period,
                tf.ftd_day_of_week,
                tf.ftd_is_weekend,
                tf.ftd_month,
                tf.ftd_hour,
                tf.registration_to_ftd_hours,
                tf.install_to_ftd_hours,

                -- Target
                m.is_vip

            FROM users_metrics AS m
                     LEFT JOIN user_time_features AS tf ON m.id = tf.user_metrics_id
                     LEFT JOIN user_financial_features AS ff ON m.id = ff.user_metrics_id
            WHERE m.is_vip IS NOT null
              AND m.use_for_training = 1;
            """


def prepare_dataset():
    session = SessionLocal()

    try:
        query = text(SQL_QUERY)

        df = pd.read_sql_query(
            query,
            session.connection(),
        )

        if df.empty:
            print("No data found.")
            return

        # Target must contain only 0 and 1
        df["is_vip"] = df["is_vip"].astype(int)

        if not set(df["is_vip"].unique()).issubset({0, 1}):
            raise ValueError("is_vip must contain only 0 and 1")

        # Create output directory
        output_dir = Path("data")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save dataset
        output_path = output_dir / "model_dataset.parquet"

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
