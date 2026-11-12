
from db.models import UsersMetrics, UserTimeFeatures
from db.database import SessionLocal


def create_time_features(user):

    ftd = user.ftd_date
    registration = user.registration_datetime
    install = user.app_install_date

    features = UserTimeFeatures(
        user_metrics_id=user.id,
        uid=user.uid,
    )

    # Calendar features
    if ftd is not None:
        day = ftd.day

        if day <= 10:
            features.ftd_month_period = "beginning"
        elif day <= 20:
            features.ftd_month_period = "middle"
        else:
            features.ftd_month_period = "end"

        features.ftd_day_of_month = day
        features.ftd_day_of_week = ftd.weekday()
        features.ftd_is_weekend = int(ftd.weekday() >= 5)
        features.ftd_month = ftd.month
        features.ftd_hour = ftd.hour

    # Registration to FTD
    if ftd is not None and registration is not None:
        features.registration_to_ftd_hours = (ftd - registration).total_seconds() / 3600

    # App install to FTD
    if ftd is not None and install is not None:
        features.install_to_ftd_hours = (ftd - install).total_seconds() / 3600

    return features


def process_users():

    session = SessionLocal()

    try:
        users = (
            session.query(UsersMetrics)
            .outerjoin(
                UserTimeFeatures,
                UsersMetrics.id == UserTimeFeatures.user_metrics_id,
            )
            .filter(UserTimeFeatures.id.is_(None))
            .all()
        )

        features = [create_time_features(user) for user in users]

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
