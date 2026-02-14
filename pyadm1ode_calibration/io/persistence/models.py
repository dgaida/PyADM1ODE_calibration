"""Database models module."""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Text, Boolean, ForeignKey, Index, JSON
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class Plant(Base):
    """
    Biogas plant configuration and metadata.
    """

    __tablename__ = "plants"

    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    location = Column(String(200))
    operator = Column(String(100))
    V_liq = Column(Float)  # Liquid volume [m3]
    V_gas = Column(Float)  # Gas volume [m3]
    T_ad = Column(Float)  # Operating temperature [K]
    P_el_nom = Column(Float)  # Nominal electrical power [kW]
    configuration = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    measurements = relationship("Measurement", back_populates="plant", cascade="all, delete-orphan")
    simulations = relationship("Simulation", back_populates="plant", cascade="all, delete-orphan")
    calibrations = relationship("Calibration", back_populates="plant", cascade="all, delete-orphan")
    substrates = relationship("Substrate", back_populates="plant", cascade="all, delete-orphan")


class Measurement(Base):
    """
    Historical measurement data for a plant.
    """

    __tablename__ = "measurements"

    id = Column(Integer, primary_key=True)
    plant_id = Column(String(50), ForeignKey("plants.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    source = Column(String(50), default="SCADA")

    # Key measurements
    Q_gas = Column(Float)
    Q_ch4 = Column(Float)
    Q_co2 = Column(Float)
    pH = Column(Float)
    VFA = Column(Float)
    TAC = Column(Float)
    T_digester = Column(Float)

    # Generic values for other parameters
    values = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    plant = relationship("Plant", back_populates="measurements")

    __table_args__ = (Index("idx_plant_timestamp", "plant_id", "timestamp"),)


class Simulation(Base):
    """
    Simulation metadata and aggregate results.
    """

    __tablename__ = "simulations"

    id = Column(String(100), primary_key=True)
    plant_id = Column(String(50), ForeignKey("plants.id"), nullable=False)
    name = Column(String(100))
    description = Column(Text)
    scenario = Column(String(50), default="baseline")
    duration = Column(Float)  # Duration in days
    parameters = Column(JSON)  # Calibrated parameters used

    # Summary metrics
    avg_Q_gas = Column(Float)
    avg_Q_ch4 = Column(Float)
    avg_CH4_content = Column(Float)
    avg_pH = Column(Float)
    avg_VFA = Column(Float)
    total_energy = Column(Float)

    status = Column(String(20), default="pending")
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    plant = relationship("Plant", back_populates="simulations")
    time_series = relationship("SimulationTimeSeries", back_populates="simulation", cascade="all, delete-orphan")


class SimulationTimeSeries(Base):
    """
    Time-series results from a simulation run.
    """

    __tablename__ = "simulation_time_series"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(String(100), ForeignKey("simulations.id"), nullable=False)
    time = Column(Float, nullable=False)  # Time in days

    # Simulated outputs
    Q_gas = Column(Float)
    Q_ch4 = Column(Float)
    Q_co2 = Column(Float)
    CH4_content = Column(Float)
    pH = Column(Float)
    VFA = Column(Float)
    TAC = Column(Float)

    simulation = relationship("Simulation", back_populates="time_series")

    __table_args__ = (Index("idx_sim_time", "simulation_id", "time"),)


class Calibration(Base):
    """
    Calibration run metadata and results.
    """

    __tablename__ = "calibrations"

    id = Column(Integer, primary_key=True)
    plant_id = Column(String(50), ForeignKey("plants.id"), nullable=False)
    calibration_type = Column(String(20))  # initial, online
    method = Column(String(50))  # de, nelder-mead, etc.
    parameters = Column(JSON)  # Optimized parameter values
    objective_value = Column(Float)
    objectives = Column(JSON)  # List of target outputs
    validation_metrics = Column(JSON)  # RMSE, R2, etc.

    data_start = Column(DateTime)
    data_end = Column(DateTime)
    success = Column(Boolean, default=True)
    message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    plant = relationship("Plant", back_populates="calibrations")


class Substrate(Base):
    """
    Substrate laboratory analysis data.
    """

    __tablename__ = "substrates"

    id = Column(Integer, primary_key=True)
    plant_id = Column(String(50), ForeignKey("plants.id"), nullable=False)
    substrate_name = Column(String(100), nullable=False)
    substrate_type = Column(String(50))
    sample_date = Column(DateTime, nullable=False)
    sample_id = Column(String(50))

    # Laboratory values
    TS = Column(Float)  # Total Solids [%]
    VS = Column(Float)  # Volatile Solids [%]
    oTS = Column(Float)  # Organic Total Solids [%]
    foTS = Column(Float)  # Fermentable oTS [%]
    RP = Column(Float)  # Raw Protein [%]
    RL = Column(Float)  # Raw Lipids [%]
    RF = Column(Float)  # Raw Fiber [%]
    NDF = Column(Float)
    ADF = Column(Float)
    ADL = Column(Float)
    pH = Column(Float)
    NH4_N = Column(Float)
    TAC = Column(Float)
    COD_S = Column(Float)  # Soluble COD
    BMP = Column(Float)  # Biomethane potential
    C_to_N = Column(Float)

    lab_name = Column(String(100))
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    plant = relationship("Plant", back_populates="substrates")
