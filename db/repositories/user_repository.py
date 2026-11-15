from typing import cast
import pandas as pd
import numpy as np

from db.database import SessionLocal
from db.enums import TrainingDatasetType

from sqlalchemy import MetaData, Table, func, text
from sqlalchemy.dialects.postgresql import insert as pg_insert


class UserRepository:
    @staticmethod
    def _validate_dataset_type(dataset_type: TrainingDatasetType) -> TrainingDatasetType:
        try:
            return TrainingDatasetType(dataset_type)
        except (TypeError, ValueError) as error:
            allowed = ", ".join(str(item.value) for item in TrainingDatasetType)
            raise ValueError(f"Unsupported dataset number; expected one of: {allowed}") from error

    @staticmethod
    def get_training_dataset(dataset_type: TrainingDatasetType) -> pd.DataFrame:
        """Получить тренировочный датасет (использует user_training_types)."""
        dataset_type = UserRepository._validate_dataset_type(dataset_type)

        query = """
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
                         JOIN user_training_types AS utt ON m.id = utt.user_id
                         LEFT JOIN user_time_features AS tf ON m.id = tf.user_metrics_id
                         LEFT JOIN user_financial_features AS ff ON m.id = ff.user_metrics_id
                WHERE m.is_vip IS NOT NULL
                  AND utt.type_id = :dataset_type \
                """

        with SessionLocal() as session:
            df = pd.read_sql_query(
                text(query),
                session.connection(),
                params={"dataset_type": int(dataset_type)},
            )
            return cast(pd.DataFrame, df)

    @staticmethod
    def get_test_dataset(dataset_type: TrainingDatasetType) -> pd.DataFrame:
        """Получить тестовый датасет (использует user_test_types)."""
        dataset_type = UserRepository._validate_dataset_type(dataset_type)

        query = """
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
                         JOIN user_test_types AS utt ON m.id = utt.user_id
                         LEFT JOIN user_time_features AS tf ON m.id = tf.user_metrics_id
                         LEFT JOIN user_financial_features AS ff ON m.id = ff.user_metrics_id
                WHERE m.is_vip IS NOT NULL
                  AND utt.type_id = :dataset_type \
                """

        with SessionLocal() as session:
            df = pd.read_sql_query(
                text(query),
                session.connection(),
                params={"dataset_type": int(dataset_type)},
            )
            return cast(pd.DataFrame, df)

    @staticmethod
    def _build_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Рассчитать временные признаки для пользователей."""
        df = df.copy()

        date_columns = [
            "ftd_date",
            "registration_datetime",
            "app_install_date",
        ]
        for column in date_columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

        ftd = df["ftd_date"]
        registration = df["registration_datetime"]
        install = df["app_install_date"]

        features = pd.DataFrame({"id": df["id"]})

        # INSTALL FEATURES

        features["has_install"] = install.notna()

        features["install_before_registration"] = (
                install.notna() & registration.notna() & (install < registration)
        )

        features["install_after_registration"] = (
                install.notna() & registration.notna() & (install > registration)
        )

        features["install_same_day_registration"] = (
                install.notna()
                & registration.notna()
                & install.dt.normalize().eq(registration.dt.normalize())
        )

        # Временные интервалы между установкой, регистрацией и FTD.
        features["install_to_registration_hours"] = (
                (registration - install).dt.total_seconds() / 3600
        )
        features["install_to_ftd_hours"] = (
                (ftd - install).dt.total_seconds() / 3600
        )
        features["install_to_ftd_days"] = (
                (ftd - install).dt.total_seconds() / 86400
        )

        # REGISTRATION -> FTD FEATURES

        registration_to_ftd = (ftd - registration).dt.total_seconds() / 3600

        features["registration_to_ftd_minutes"] = registration_to_ftd * 60
        features["registration_to_ftd_hours"] = registration_to_ftd
        features["registration_to_ftd_days"] = registration_to_ftd / 24

        # Время до первого депозита. Если даты отсутствуют, признак будет NULL.
        features["ftd_within_1_hour"] = (
            registration_to_ftd.between(0, 1)
            .where(registration_to_ftd.notna())
            .astype("boolean")
        )
        features["ftd_within_24_hours"] = (
            registration_to_ftd.between(0, 24)
            .where(registration_to_ftd.notna())
            .astype("boolean")
        )
        features["ftd_within_3_days"] = (
            registration_to_ftd.between(0, 72)
            .where(registration_to_ftd.notna())
            .astype("boolean")
        )
        features["ftd_within_7_days"] = (
            registration_to_ftd.between(0, 168)
            .where(registration_to_ftd.notna())
            .astype("boolean")
        )

        features["registration_to_ftd_same_day"] = (
                registration.notna()
                & ftd.notna()
                & registration.dt.normalize().eq(ftd.dt.normalize())
        )

        # CALENDAR FEATURES

        date_map = {
            "registration": registration,
            "install": install,
            "ftd": ftd,
        }

        for prefix, dates in date_map.items():
            hour = dates.dt.hour
            day = dates.dt.day

            # День недели: воскресенье = 0, понедельник = 1, ..., суббота = 6.
            weekday = (dates.dt.dayofweek + 1) % 7

            features[f"{prefix}_hour"] = hour.astype("Int64")
            features[f"{prefix}_day_of_week"] = weekday.astype("Int64")
            features[f"{prefix}_day_of_month"] = day.astype("Int64")
            features[f"{prefix}_month"] = dates.dt.month.astype("Int64")

            features[f"{prefix}_is_weekend"] = (
                weekday.isin([0, 6])
                .where(dates.notna())
                .astype("boolean")
            )

            # Период месяца: начало (1-10), середина (11-20), конец (21-31).
            month_period = pd.Series(None, index=df.index, dtype="object")
            month_period.loc[day.between(1, 10)] = "beginning"
            month_period.loc[day.between(11, 20)] = "middle"
            month_period.loc[day.between(21, 31)] = "end"

            features[f"{prefix}_month_period"] = month_period

            # Период суток.
            day_period = pd.Series(None, index=df.index, dtype="object")
            day_period.loc[hour.between(0, 5)] = "night"
            day_period.loc[hour.between(6, 11)] = "morning"
            day_period.loc[hour.between(12, 17)] = "afternoon"
            day_period.loc[hour.between(18, 23)] = "evening"

            features[f"{prefix}_day_period"] = day_period

            # Циклическое кодирование часа: 23:00 близок к 00:00.
            features[f"{prefix}_hour_sin"] = np.sin(2 * np.pi * hour / 24)
            features[f"{prefix}_hour_cos"] = np.cos(2 * np.pi * hour / 24)

            # Циклическое кодирование дня недели.
            features[f"{prefix}_weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
            features[f"{prefix}_weekday_cos"] = np.cos(2 * np.pi * weekday / 7)

        return features

    @staticmethod
    def refresh_time_features(batch_size: int = 500) -> int:
        """Рассчитать временные признаки и обновить таблицу vip_features."""

        query = """
                SELECT id,
                       ftd_date,
                       registration_datetime,
                       app_install_date
                FROM users_metrics
                """

        with SessionLocal() as session:
            try:
                # Получаем исходные данные.
                df = pd.read_sql_query(
                    text(query),
                    session.connection(),
                )

                if df.empty:
                    return 0

                # Рассчитываем временные признаки.
                features = UserRepository._build_time_features(df)

                # Преобразуем NaN, NaT и pd.NA в None для PostgreSQL.
                records = features.astype(object).where(
                    pd.notna(features), None,
                ).to_dict(orient="records")

                # Получаем описание целевой таблицы из PostgreSQL.
                metadata = MetaData()
                table = Table(
                    "vip_features",
                    metadata,
                    autoload_with=session.connection(),
                )

                # При конфликте обновляем признаки, но не исходный id и created_at.
                update_columns = {
                    column.name: getattr(
                        pg_insert(table).excluded, column.name,
                    )
                    for column in table.columns
                    if column.name not in {"id", "created_at", "updated_at"}
                }
                update_columns["updated_at"] = func.now()

                # Записываем данные пакетами.
                for start in range(0, len(records), batch_size):
                    batch = records[start:start + batch_size]

                    statement = pg_insert(table).values(batch)
                    statement = statement.on_conflict_do_update(
                        index_elements=[table.c.id],
                        set_=update_columns,
                    )

                    session.execute(statement)

                session.commit()

                return len(features)

            except Exception:
                session.rollback()
                raise

    @staticmethod
    def get_test_dataset_without_finances(dataset_type: TrainingDatasetType) -> pd.DataFrame:
        """Получить тестовый датасет без финансовых признаков."""
        dataset_type = UserRepository._validate_dataset_type(dataset_type)

        query = """
                SELECT
                    -- Categorical feature
                    m.email_domain,

                    -- Installation features
                    tf.has_install,
                    tf.install_before_registration,
                    tf.install_after_registration,
                    tf.install_same_day_registration,
                    tf.install_to_registration_hours,
                    tf.install_to_ftd_hours,
                    tf.install_to_ftd_days,

                    -- Registration -> FTD features
                    tf.registration_to_ftd_minutes,
                    tf.registration_to_ftd_hours,
                    tf.registration_to_ftd_days,
                    tf.ftd_within_1_hour,
                    tf.ftd_within_24_hours,
                    tf.ftd_within_3_days,
                    tf.ftd_within_7_days,
                    tf.registration_to_ftd_same_day,

                    -- Registration time features
                    tf.registration_hour,
                    tf.registration_day_of_week,
                    tf.registration_day_of_month,
                    tf.registration_month,
                    tf.registration_is_weekend,
                    tf.registration_month_period,
                    tf.registration_day_period,
                    tf.registration_hour_sin,
                    tf.registration_hour_cos,
                    tf.registration_weekday_sin,
                    tf.registration_weekday_cos,

                    -- Installation time features
                    tf.install_hour,
                    tf.install_day_of_week,
                    tf.install_day_of_month,
                    tf.install_month,
                    tf.install_is_weekend,
                    tf.install_month_period,
                    tf.install_day_period,
                    tf.install_hour_sin,
                    tf.install_hour_cos,
                    tf.install_weekday_sin,
                    tf.install_weekday_cos,

                    -- FTD time features
                    tf.ftd_day_of_month,
                    tf.ftd_month_period,
                    tf.ftd_day_of_week,
                    tf.ftd_is_weekend,
                    tf.ftd_month,
                    tf.ftd_hour,
                    tf.ftd_day_period,
                    tf.ftd_hour_sin,
                    tf.ftd_hour_cos,
                    tf.ftd_weekday_sin,
                    tf.ftd_weekday_cos,

                    -- Target
                    m.is_vip

                FROM users_metrics AS m
                         JOIN user_test_types AS utt
                              ON m.id = utt.user_id
                         LEFT JOIN vip_features AS tf
                                   ON m.id = tf.id
                WHERE m.is_vip IS NOT NULL
                  AND utt.type_id = :dataset_type
                """

        with SessionLocal() as session:
            df = pd.read_sql_query(
                text(query),
                session.connection(),
                params={"dataset_type": int(dataset_type)},
            )
            return cast(pd.DataFrame, df)

    @staticmethod
    def get_training_dataset_without_finances(dataset_type: TrainingDatasetType) -> pd.DataFrame:
        """Получить тренировочный датасет без финансовых признаков."""
        dataset_type = UserRepository._validate_dataset_type(dataset_type)

        query = """
                SELECT
                    -- Categorical feature
                    m.email_domain,

                    -- Installation features
                    tf.has_install,
                    tf.install_before_registration,
                    tf.install_after_registration,
                    tf.install_same_day_registration,
                    tf.install_to_registration_hours,
                    tf.install_to_ftd_hours,
                    tf.install_to_ftd_days,

                    -- Registration -> FTD features
                    tf.registration_to_ftd_minutes,
                    tf.registration_to_ftd_hours,
                    tf.registration_to_ftd_days,
                    tf.ftd_within_1_hour,
                    tf.ftd_within_24_hours,
                    tf.ftd_within_3_days,
                    tf.ftd_within_7_days,
                    tf.registration_to_ftd_same_day,

                    -- Registration time features
                    tf.registration_hour,
                    tf.registration_day_of_week,
                    tf.registration_day_of_month,
                    tf.registration_month,
                    tf.registration_is_weekend,
                    tf.registration_month_period,
                    tf.registration_day_period,
                    tf.registration_hour_sin,
                    tf.registration_hour_cos,
                    tf.registration_weekday_sin,
                    tf.registration_weekday_cos,

                    -- Installation time features
                    tf.install_hour,
                    tf.install_day_of_week,
                    tf.install_day_of_month,
                    tf.install_month,
                    tf.install_is_weekend,
                    tf.install_month_period,
                    tf.install_day_period,
                    tf.install_hour_sin,
                    tf.install_hour_cos,
                    tf.install_weekday_sin,
                    tf.install_weekday_cos,

                    -- FTD time features
                    tf.ftd_day_of_month,
                    tf.ftd_month_period,
                    tf.ftd_day_of_week,
                    tf.ftd_is_weekend,
                    tf.ftd_month,
                    tf.ftd_hour,
                    tf.ftd_day_period,
                    tf.ftd_hour_sin,
                    tf.ftd_hour_cos,
                    tf.ftd_weekday_sin,
                    tf.ftd_weekday_cos,

                    -- Target
                    m.is_vip

                FROM users_metrics AS m
                         JOIN user_training_types AS utt
                              ON m.id = utt.user_id
                         LEFT JOIN vip_features AS tf
                                   ON m.id = tf.id
                WHERE m.is_vip IS NOT NULL
                  AND utt.type_id = :dataset_type \
                """

        with SessionLocal() as session:
            df = pd.read_sql_query(
                text(query),
                session.connection(),
                params={"dataset_type": int(dataset_type)},
            )
            return cast(pd.DataFrame, df)

    @staticmethod
    def get_training_dataset_top_fields(dataset_type: TrainingDatasetType) -> pd.DataFrame:
        """Получить тренировочный датасет (использует user_training_types)."""
        dataset_type = UserRepository._validate_dataset_type(dataset_type)

        query = """
                SELECT
                    ff.deposits_sum_ftd_day,
                    ff.avg_transaction_check,
                    ff.transactions_count_ftd_day,
                    ff.spins_count_ftd_day,
                    ff.avg_bet_size_per_spin,

                    m.email_domain,

                    tf.ftd_hour,
                    tf.registration_to_ftd_hours,
                    tf.ftd_day_of_month,
                    tf.ftd_day_of_week,
                    tf.ftd_month,
                    tf.install_to_ftd_hours,

                    -- Target
                    m.is_vip
                FROM users_metrics AS m
                         JOIN user_training_types AS utt ON m.id = utt.user_id
                         LEFT JOIN user_time_features AS tf ON m.id = tf.user_metrics_id
                         LEFT JOIN user_financial_features AS ff ON m.id = ff.user_metrics_id
                WHERE m.is_vip IS NOT NULL
                  AND utt.type_id = :dataset_type \
                """

        with SessionLocal() as session:
            df = pd.read_sql_query(
                text(query),
                session.connection(),
                params={"dataset_type": int(dataset_type)},
            )
            return cast(pd.DataFrame, df)

    @staticmethod
    def get_test_dataset_top_fields(dataset_type: TrainingDatasetType) -> pd.DataFrame:
        """Получить тестовый датасет (использует user_test_types)."""
        dataset_type = UserRepository._validate_dataset_type(dataset_type)

        query = """
                SELECT
                    ff.deposits_sum_ftd_day,
                    ff.avg_transaction_check,
                    ff.transactions_count_ftd_day,
                    ff.spins_count_ftd_day,
                    ff.avg_bet_size_per_spin,

                    m.email_domain,

                    tf.ftd_hour,
                    tf.registration_to_ftd_hours,
                    tf.ftd_day_of_month,
                    tf.ftd_day_of_week,
                    tf.ftd_month,
                    tf.install_to_ftd_hours,

                    -- Target
                    m.is_vip
                FROM users_metrics AS m
                         JOIN user_test_types AS utt ON m.id = utt.user_id
                         LEFT JOIN user_time_features AS tf ON m.id = tf.user_metrics_id
                         LEFT JOIN user_financial_features AS ff ON m.id = ff.user_metrics_id
                WHERE m.is_vip IS NOT NULL
                  AND utt.type_id = :dataset_type
                """

        with SessionLocal() as session:
            df = pd.read_sql_query(
                text(query),
                session.connection(),
                params={"dataset_type": int(dataset_type)},
            )
            return cast(pd.DataFrame, df)
