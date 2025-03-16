from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, ForeignKey, Boolean, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# ✅ PostgreSQL Connection String
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/medscan_app_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ✅ Define the Physician Table for Signup
class Physician(Base):
    __tablename__ = "physicians"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    physician_name = Column(String, nullable=False)
    medical_title = Column(String, nullable=False)
    phy_hospital_clinic = Column(String, nullable=False)  # Updated column name
    email = Column(String, unique=True, nullable=False)  # Physician's email (unique)
    password = Column(String, nullable=False)  # Hashed password storage

# ✅ Define the Patient Table (Prepopulated Fields Only)
class PatientRecord(Base):
    __tablename__ = "patient_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id = Column(String, unique=True, nullable=False)  # Unique Patient ID
    patient_email = Column(String, nullable=False)  # ✅ Changed from user_email
    patient_name = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)
    symptoms = Column(Text, nullable=True)
    image_path = Column(Text, nullable=False)
    referring_doctor = Column(String, nullable=True)  # ✅ Newly added field (Prepopulated)
    referring_doctor_clinic = Column(String, nullable=True)  # ✅ Newly added field (Prepopulated)
    case_processed = Column(Boolean, default=False)  # ✅ Initially 0, updated to 1 on signoff
    created_at = Column(TIMESTAMP, server_default=func.now())

# ✅ Define the Patient Predictions Table (Updated After Signoff)
class PatientPrediction(Base):
    __tablename__ = "patient_predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id = Column(String, ForeignKey("patient_records.patient_id"), nullable=False)
    physician_name = Column(String, nullable=False)  # ✅ Updated after signoff
    physician_email = Column(String, nullable=False)  # ✅ Updated after signoff
    medical_title = Column(String, nullable=False)  # ✅ Updated after signoff
    phy_hospital_clinic = Column(String, nullable=False)  # ✅ Updated after signoff
    prediction = Column(Text, nullable=False)
    prescription_report_ai_gen = Column(Text, nullable=True)  # ✅ AI-Generated Report
    prescription_report_physician = Column(Text, nullable=True)  # ✅ Physician's Manual Report
    case_processed = Column(Boolean, default=False)  # ✅ Initially 0, updated to 1 on signoff
    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationship with PatientRecord (for querying purposes)
    patient = relationship("PatientRecord", backref="predictions")

# ✅ Create Tables in the Database
Base.metadata.create_all(bind=engine)

# ✅ Function to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
