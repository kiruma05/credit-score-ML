from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Numeric, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from pydantic import BaseModel
from typing import Dict, Any, Optional
from .database import Base
from datetime import datetime


# ==============================================================================
# --- SQLAlchemy DB Models (How data is stored in PostgreSQL) ---
# ==============================================================================

class Customer(Base):
    __tablename__ = "customers"
    id = Column(Integer, primary_key=True)
    customer_id = Column(String, unique=True, index=True, nullable=False)
    nida = Column(String, unique=True, index=True, nullable=False)
    first_name = Column(String)
    surname = Column(String)
    date_of_birth = Column(Date)

    inferences = relationship("CachedInference", back_populates="customer", cascade="all, delete-orphan")
    loans = relationship("Loan", back_populates="customer", cascade="all, delete-orphan")


class CachedInference(Base):
    __tablename__ = "cached_inferences"
    id = Column(Integer, primary_key=True)
    customer_id = Column(String, ForeignKey("customers.customer_id"), nullable=False)

    credit_score = Column(Float)
    decision = Column(String)
    recommended_credit_limit = Column(Numeric(12, 2))
    suggested_interest_rate = Column(Float)
    risk_category = Column(String)
    risk_probability = Column(Float)

    validity_period_days = Column(Integer)
    last_inference_date = Column(Date, nullable=False)
    end_inference_date = Column(Date, nullable=False, index=True)
    model_version = Column(String)

    customer = relationship("Customer", back_populates="inferences")
    loan = relationship("Loan", back_populates="inference", uselist=False)


class Loan(Base):
    __tablename__ = "loans"
    id = Column(Integer, primary_key=True)
    loan_ref = Column(String, unique=True, index=True, nullable=False)
    customer_id = Column(String, ForeignKey("customers.customer_id"), nullable=False)
    inference_id = Column(Integer, ForeignKey("cached_inferences.id"), unique=False)

    disbursed_amount = Column(Numeric(12, 2), nullable=False)
    outstanding_balance = Column(Numeric(12, 2), nullable=False)
    disbursal_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="ACTIVE")

    customer = relationship("Customer", back_populates="loans")
    inference = relationship("CachedInference", back_populates="loan")
    repayments = relationship("Repayment", back_populates="loan", cascade="all, delete-orphan")


class Repayment(Base):
    __tablename__ = "repayments"
    id = Column(Integer, primary_key=True)
    loan_ref = Column(String, ForeignKey("loans.loan_ref"), nullable=False)
    amount = Column(Numeric(12, 2), nullable=False)
    payment_date = Column(DateTime, default=datetime.utcnow)

    loan = relationship("Loan", back_populates="repayments")


class ApiClient(Base):
    __tablename__ = "api_clients"
    id = Column(Integer, primary_key=True)
    client_name = Column(String, unique=True, index=True, nullable=False)
    api_key = Column(String, unique=True, index=True, nullable=False)
    api_secret_hash = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime, nullable=True)       # null = never expires (dev-safe)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    scopes = Column(String, default="*")               # reserved for future scope-based auth


# ==============================================================================
# --- Pydantic API Models (How data looks in API requests/responses) ---
# ==============================================================================

class PredictionRequest(BaseModel):
    nida: str


class PredictionResponse(BaseModel):
    customer_id: str
    request_id: str
    timestamp: str
    credit_score: float
    risk_category: str
    risk_probability: float
    decision: str
    recommended_credit_limit: float
    currency: str = "TZS"
    spending_limit: float
    outstanding_balance: float
    suggested_interest_rate: float
    validity_period: str
    model_version: str


class DisburseRequest(BaseModel):
    customer_id: str
    loan_ref: str
    amount: float


class RepayRequest(BaseModel):
    customer_id: str
    loan_ref: str
    amount: float


class StatusResponse(BaseModel):
    message: str
    status: str
    detail: Optional[Dict[str, Any]] = None