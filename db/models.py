from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    UniqueConstraint, func,
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class UsersMetrics(Base):
    __tablename__ = "users_metrics"

    id = Column(Integer, primary_key=True)
    uid = Column(String(36), nullable=False, unique=True)

    ftd_date = Column(DateTime)
    registration_datetime = Column(DateTime)
    app_install_date = Column(DateTime)

    avg_bet_size_per_spin = Column(Float)
    spins_count_ftd_day = Column(Float)
    email_domain = Column(String)

    failed_deposits_count = Column(Integer)
    deposits_sum_ftd_day = Column(Float)
    transactions_count_ftd_day = Column(Integer)
    avg_transaction_check = Column(Float)

    is_vip = Column(Integer)
    dep_60_days = Column(Float)
    ltv = Column(Float)

    time_features = relationship(
        "UserTimeFeatures",
        back_populates="user",
        uselist=False,
    )


class UserTimeFeatures(Base):
    __tablename__ = "user_time_features"

    id = Column(Integer, primary_key=True, autoincrement=True)

    user_metrics_id = Column(
        Integer,
        ForeignKey("users_metrics.id"),
        nullable=False,
    )

    uid = Column(String(36), nullable=False, unique=True)

    ftd_day_of_month = Column(Integer)
    ftd_month_period = Column(String(20))
    ftd_day_of_week = Column(Integer)
    ftd_is_weekend = Column(Integer)
    ftd_month = Column(Integer)
    ftd_hour = Column(Integer)

    registration_to_ftd_hours = Column(Float)
    install_to_ftd_hours = Column(Float)

    user = relationship(
        "UsersMetrics",
        back_populates="time_features",
    )

    __table_args__ = (
        UniqueConstraint(
            "user_metrics_id",
            name="uq_time_features_user_metrics",
        ),
    )


class UserFinancialFeatures(Base):
    __tablename__ = "user_financial_features"

    id = Column(Integer, primary_key=True, index=True)

    user_metrics_id = Column(
        Integer,
        ForeignKey("users_metrics.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    uid = Column(String(36), nullable=False, unique=True)

    avg_bet_size_per_spin = Column(Float)
    spins_count_ftd_day = Column(Float)

    failed_deposits_count = Column(Integer)
    deposits_sum_ftd_day = Column(Float)
    transactions_count_ftd_day = Column(Integer)
    avg_transaction_check = Column(Float)

    created_at = Column(DateTime, server_default=func.now())
