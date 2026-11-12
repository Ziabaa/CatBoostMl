# Структура базы данных

База данных состоит из трёх таблиц:

1. `users_metrics` — исходные данные пользователей.
2. `user_time_features` — временные признаки, рассчитанные на основе исходных данных.
3. `user_financial_features` — финансовые и игровые признаки на конец FTD day.

## 1. Таблица `users_metrics`

**Назначение:** хранение исходных данных о пользователях, полученных из источника. Содержит целевую переменную и данные, на основе которых формируются признаки.

| Колонка | Тип данных | Описание |
|---|---|---|
| `id` | SERIAL | Уникальный внутренний ID записи, первичный ключ |
| `uid` | TEXT | Идентификатор пользователя во внешней системе |
| `ftd_date` | TIMESTAMP | Дата и время первого депозита (FTD) |
| `registration_datetime` | TIMESTAMP | Дата и время регистрации пользователя |
| `avg_bet_size_per_spin` | FLOAT | Средний размер ставки за один спин в FTD day |
| `spins_count_ftd_day` | FLOAT | Количество спинов в FTD day |
| `email_domain` | TEXT | Домен электронной почты пользователя |
| `app_install_date` | TIMESTAMP | Дата и время установки приложения |
| `failed_deposits_count` | INT | Количество неудачных депозитов в FTD day |
| `deposits_sum_ftd_day` | FLOAT | Общая сумма депозитов в FTD day |
| `transactions_count_ftd_day` | INT | Количество транзакций в FTD day |
| `avg_transaction_check` | FLOAT | Средний чек транзакции в FTD day |
| `is_vip` | INT | Целевая переменная: VIP (1) или не VIP (0) |
| `dep_60_days` | FLOAT | Сумма депозитов за 60 дней |
| `ltv` | FLOAT | Lifetime Value — общая ценность пользователя |

## 2. Таблица `user_time_features`

**Назначение:** хранение временных признаков, рассчитанных на основе дат из `users_metrics`. Используется для выявления закономерностей, связанных со временем регистрации, установки приложения и первого депозита.

| Колонка | Тип данных | Описание |
|---|---|---|
| `id` | SERIAL | Уникальный внутренний ID записи, первичный ключ |
| `user_metrics_id` | INTEGER | Ссылка на исходную запись в `users_metrics` |
| `uid` | VARCHAR(36) | Идентификатор пользователя |
| `ftd_day_of_month` | INTEGER | День месяца, когда произошёл FTD |
| `ftd_month_period` | VARCHAR(20) | Период месяца: `beginning` (1–10), `middle` (11–20), `end` (21–31) |
| `ftd_day_of_week` | INTEGER | День недели FTD (0 — понедельник, 6 — воскресенье) |
| `ftd_is_weekend` | INTEGER | Признак выходного дня: 1 — выходной, 0 — будний день |
| `ftd_month` | INTEGER | Месяц, когда произошёл FTD (1–12) |
| `ftd_hour` | INTEGER | Час совершения FTD (0–23) |
| `registration_to_ftd_hours` | DOUBLE PRECISION | Количество часов между регистрацией и первым депозитом |
| `install_to_ftd_hours` | DOUBLE PRECISION | Количество часов между установкой приложения и первым депозитом |
| `created_at` | TIMESTAMP | Дата и время создания записи с признаками |

## 3. Таблица `user_financial_features`

**Назначение:** хранение финансовых и игровых признаков пользователя, рассчитанных по данным, доступным на конец FTD day. Используется для анализа первоначальной активности пользователя.

| Колонка | Тип данных | Описание |
|---|---|---|
| `id` | SERIAL | Уникальный внутренний ID записи, первичный ключ |
| `user_metrics_id` | INTEGER | Ссылка на исходную запись в `users_metrics` |
| `uid` | VARCHAR(36) | Идентификатор пользователя |
| `avg_bet_size_per_spin` | DOUBLE PRECISION | Средний размер ставки за один спин |
| `spins_count_ftd_day` | DOUBLE PRECISION | Количество спинов в FTD day |
| `failed_deposits_count` | INTEGER | Количество неудачных депозитов в FTD day |
| `deposits_sum_ftd_day` | DOUBLE PRECISION | Общая сумма депозитов в FTD day |
| `transactions_count_ftd_day` | INTEGER | Количество транзакций в FTD day |
| `avg_transaction_check` | DOUBLE PRECISION | Средний чек транзакции в FTD day |
| `created_at` | TIMESTAMP | Дата и время создания записи с признаками |

## 4. Связи между таблицами

Связи организованы через колонку `user_metrics_id`.

```text
                 users_metrics
                 -------------
                 id (PK)
                 uid
                    |
          __________|__________
         |                     |
         ▼                     ▼
  user_time_features   user_financial_features
  ------------------   -----------------------
  id (PK)              id (PK)
  user_metrics_id (FK) user_metrics_id (FK)
  uid                  uid
  time features        financial features
```

**Тип связей:**

- `users_metrics` → `user_time_features`: один к одному (1:1).
- `users_metrics` → `user_financial_features`: один к одному (1:1).

**Ограничения и зависимости:**

| Ограничение | Описание |
|---|---|
| `PRIMARY KEY (id)` | Уникальный идентификатор записи внутри каждой таблицы |
| `FOREIGN KEY (user_metrics_id)` | Связь с исходной записью в `users_metrics` |
| `UNIQUE (user_metrics_id)` | Не позволяет создать несколько наборов признаков одного типа для одной исходной записи |
| `UNIQUE (uid)` | Не позволяет хранить дублирующийся UID внутри одной таблицы |
| `ON DELETE CASCADE` | При удалении исходной записи из `users_metrics` связанные записи признаков удаляются автоматически |

## 5. Общая логика использования данных

- `users_metrics` — источник данных и целевая переменная.
- `user_time_features` — временные признаки.
- `user_financial_features` — финансовые и игровые признаки.

Для обучения CatBoost таблицы признаков объединяются по `user_metrics_id`.

**Важно для ML:** `is_vip` используется как целевая переменная. `dep_60_days` и `ltv` не включаются в признаки, поскольку могут содержать информацию о поведении пользователя после FTD day.
