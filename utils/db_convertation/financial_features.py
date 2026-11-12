from db.models import UsersMetrics, UserFinancialFeatures
from db.database import SessionLocal


def create_financial_features(user):
    return UserFinancialFeatures(
        user_metrics_id=user.id,
        uid=user.uid,

        avg_bet_size_per_spin=user.avg_bet_size_per_spin,
        spins_count_ftd_day=user.spins_count_ftd_day,

        failed_deposits_count=user.failed_deposits_count,
        deposits_sum_ftd_day=user.deposits_sum_ftd_day,
        transactions_count_ftd_day=user.transactions_count_ftd_day,
        avg_transaction_check=user.avg_transaction_check,
    )


def process_users():
    session = SessionLocal()

    try:
        users = (
            session.query(UsersMetrics)
            .outerjoin(
                UserFinancialFeatures,
                UsersMetrics.id == UserFinancialFeatures.user_metrics_id
            )
            .filter(UserFinancialFeatures.id.is_(None))
            .all()
        )

        features = [
            create_financial_features(user)
            for user in users
        ]

        if features:
            session.add_all(features)
            session.commit()

        print(f"Processed users: {len(features)}")

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()


if __name__ == "__main__":
    process_users()