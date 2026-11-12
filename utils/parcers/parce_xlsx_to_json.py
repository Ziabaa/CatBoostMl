import pandas as pd

excel_file = "test_dataset.xlsx"
json_output_file = "user_data.json"

df = pd.read_excel(excel_file)

# convert to correct format
column_mapping = {
    "uid": "uid",
    "дата фтд": "ftd_date",
    " дата і час реєстрації": "registration_datetime",
    "середній розмір бету на спін": "avg_bet_size_per_spin",
    " к-ть спінів в день фтд": "spins_count_ftd_day",
    "поштовий домен": "email_domain",
    "дата встановлення застосунку в період до дня фтд включно": "app_install_date",
    "невдалих депозитів до дня фтд влючно": "failed_deposits_count",
    "сума депозитів в день фтд": "deposits_sum_ftd_day",
    "к-ть транзакцій в день фтд": "transactions_count_ftd_day",
    " середні чек транзакції в день фтд": "avg_transaction_check",
    "віп/не віп": "is_vip",
    "деп 60 днів": "dep_60_days",
    "ltv": "ltv",
}

df = df.rename(columns=column_mapping)

df.to_json(json_output_file, orient="records", force_ascii=False, indent=4)

print(f"[+] Файл успешно конвертирован и сохранен как '{json_output_file}'!")
