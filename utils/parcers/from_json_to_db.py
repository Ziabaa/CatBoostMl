import json
import pandas as pd
from sqlalchemy import create_engine, text

# 1. Читаем JSON-файл
json_filename = "user_data.json"
try:
    with open(json_filename, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    print(f"[1/4] Файл '{json_filename}' успешно прочитан. Строк: {len(json_data)}")
except Exception as e:
    print(f"[-] Ошибка чтения JSON: {e}")
    exit()

df = pd.DataFrame(json_data)

# 2. Конвертируем даты
date_columns = ["ftd_date", "registration_datetime", "app_install_date"]
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], unit="ms")

# 3. Подключаемся к базе данных
db_url = "postgresql://postgres:1234@localhost:5432/user_stats_db"
engine = create_engine(db_url)

# 4. Создаем таблицу и заливаем данные вручную через соединение с явным commit
try:
    with engine.begin() as connection:  # connection.begin() автоматически делает commit при успехе или rollback при ошибке

        connection.execute(text("DROP TABLE IF EXISTS users_metrics;"))

        create_table_sql = """
                           CREATE TABLE users_metrics
                           (
                               id                         SERIAL PRIMARY KEY,
                               uid                        TEXT,
                               ftd_date                   TIMESTAMP,
                               registration_datetime      TIMESTAMP,
                               avg_bet_size_per_spin      FLOAT,
                               spins_count_ftd_day        FLOAT,
                               email_domain               TEXT,
                               app_install_date           TIMESTAMP,
                               failed_deposits_count      INT,
                               deposits_sum_ftd_day       FLOAT,
                               transactions_count_ftd_day INT,
                               avg_transaction_check      FLOAT,
                               is_vip                     INT,
                               dep_60_days                FLOAT,
                               ltv                        FLOAT
                           );
                           """
        connection.execute(text(create_table_sql))
        print("[2/4] Таблица 'users_metrics' успешно создана в БД.")

    # Теперь записываем данные через pandas (используем connection/engine)
    df.to_sql("users_metrics", engine, if_exists="append", index=False)
    print(f"[3/4] Данные успешно записаны в таблицу. Всего строк: {len(df)}")

    # Проверяем, сколько строк реально в таблице
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM users_metrics;")).scalar()
        print(f"[4/4] Проверка в БД: в таблице 'users_metrics' сейчас записей -> {result}")

except Exception as e:
    print(f"\n[-] ПРОИЗОШЛА ОШИБКА ПРИ РАБОТЕ С БД:\n{e}")