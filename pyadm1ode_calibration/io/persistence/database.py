from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd
import numpy as np
from contextlib import contextmanager
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from .models import Base, Plant, Measurement, Simulation, SimulationTimeSeries, Calibration, Substrate
from .connection import ConnectionManager, DatabaseConfig
from ...exceptions import DatabaseError


class Database:
    """
    PostgreSQL database interface for PyADM1.
    """

    def __init__(self, connection_string: Optional[str] = None, config: Optional[DatabaseConfig] = None):
        self.connection_manager = ConnectionManager(connection_string, config)
        self.engine = self.connection_manager.engine
        self.SessionLocal = self.connection_manager.SessionLocal
        self.connection_string = self.connection_manager.connection_string

    @classmethod
    def from_env(cls, prefix: str = "DB") -> "Database":
        """Create database from environment variables."""
        import os
        from urllib.parse import quote_plus

        host = os.getenv(f"{prefix}_HOST", "localhost")
        port = os.getenv(f"{prefix}_PORT", "5432")
        database = os.getenv(f"{prefix}_NAME")
        username = os.getenv(f"{prefix}_USER")
        password = os.getenv(f"{prefix}_PASSWORD")

        if not all([database, username, password]):
            raise ValueError("Missing required environment variables")

        conn_str = f"postgresql://{username}:{quote_plus(password)}@{host}:{port}/{database}"
        return cls(connection_string=conn_str)

    @contextmanager
    def get_session(self) -> Session:
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_all_tables(self) -> None:
        Base.metadata.create_all(bind=self.engine)

    def drop_all_tables(self) -> None:
        Base.metadata.drop_all(bind=self.engine)

    def create_plant(
        self,
        plant_id: str,
        name: str,
        location: Optional[str] = None,
        operator: Optional[str] = None,
        V_liq: Optional[float] = None,
        V_gas: Optional[float] = None,
        T_ad: Optional[float] = None,
        P_el_nom: Optional[float] = None,
        configuration: Optional[Dict] = None,
    ) -> Plant:
        session = self.SessionLocal()
        try:
            plant = Plant(
                id=plant_id,
                name=name,
                location=location,
                operator=operator,
                V_liq=V_liq,
                V_gas=V_gas,
                T_ad=T_ad,
                P_el_nom=P_el_nom,
                configuration=configuration,
            )
            session.add(plant)
            session.commit()
            session.refresh(plant)
            session.expunge(plant)
            return plant
        except IntegrityError:
            session.rollback()
            raise ValueError(f"Plant with ID '{plant_id}' already exists")
        finally:
            session.close()

    def get_plant(self, plant_id: str) -> Plant:
        session = self.SessionLocal()
        try:
            plant = session.query(Plant).filter(Plant.id == plant_id).first()
            if plant is None:
                raise ValueError(f"Plant '{plant_id}' not found")
            session.expunge(plant)
            return plant
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to retrieve plant '{plant_id}': {e}")
        finally:
            session.close()

    def list_plants(self) -> List[Dict[str, Any]]:
        with self.get_session() as session:
            plants = session.query(Plant).all()
            return [
                {
                    "id": p.id,
                    "name": p.name,
                    "location": p.location,
                    "V_liq": p.V_liq,
                    "V_gas": p.V_gas,
                    "T_ad": p.T_ad,
                    "created_at": p.created_at,
                }
                for p in plants
            ]

    def store_measurements(self, plant_id: str, data: pd.DataFrame, source: str = "SCADA", validate: bool = True) -> int:
        self.get_plant(plant_id)
        if "timestamp" not in data.columns:
            raise ValueError("DataFrame must have 'timestamp' column")
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            data["timestamp"] = pd.to_datetime(data["timestamp"])
        if validate:
            from ..validation.validators import DataValidator

            validation = DataValidator.validate(data)
            if not validation.is_valid:
                pass
        records = []
        for _, row in data.iterrows():
            record = {"plant_id": plant_id, "timestamp": row["timestamp"], "source": source}
            for col in data.columns:
                if col != "timestamp" and col in Measurement.__table__.columns:
                    val = row[col]
                    if pd.notna(val):
                        record[col] = float(val) if isinstance(val, (int, float, np.number)) else val
            records.append(record)
        with self.get_session() as session:
            try:
                session.bulk_insert_mappings(Measurement, records)
                return len(records)
            except SQLAlchemyError as e:
                raise DatabaseError(f"Failed to store measurements: {e}")

    def load_measurements(self, plant_id: str, start_time=None, end_time=None, columns=None, source=None) -> pd.DataFrame:
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
        with self.get_session() as session:
            query = session.query(Measurement).filter(Measurement.plant_id == plant_id)
            if start_time:
                query = query.filter(Measurement.timestamp >= start_time)
            if end_time:
                query = query.filter(Measurement.timestamp <= end_time)
            if source:
                query = query.filter(Measurement.source == source)
            results = query.order_by(Measurement.timestamp).all()
            if not results:
                return pd.DataFrame()
            data_dict = {"timestamp": [r.timestamp for r in results]}
            if columns is None:
                columns = [
                    c.name
                    for c in Measurement.__table__.columns
                    if c.name not in ["id", "plant_id", "timestamp", "source", "created_at"]
                ]
            for col in columns:
                data_dict[col] = [getattr(r, col) for r in results]
            return pd.DataFrame(data_dict).set_index("timestamp")

    def store_simulation(
        self,
        simulation_id: str,
        plant_id: str,
        results: List[Dict[str, Any]],
        name: Optional[str] = None,
        description: Optional[str] = None,
        duration: Optional[float] = None,
        parameters: Optional[Dict] = None,
        scenario: str = "baseline",
    ) -> Simulation:
        self.get_plant(plant_id)
        metrics = self._calculate_simulation_metrics(results)
        with self.get_session() as session:
            sim = Simulation(
                id=simulation_id,
                plant_id=plant_id,
                name=name,
                description=description,
                duration=duration or (results[-1]["time"] if results else 0),
                scenario=scenario,
                parameters=parameters,
                avg_Q_gas=metrics.get("avg_Q_gas"),
                avg_Q_ch4=metrics.get("avg_Q_ch4"),
                avg_CH4_content=metrics.get("avg_CH4_content"),
                avg_pH=metrics.get("avg_pH"),
                avg_VFA=metrics.get("avg_VFA"),
                total_energy=metrics.get("total_energy"),
                status="completed",
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
            )
            try:
                session.add(sim)
                session.flush()
                ts_records = []
                for res in results:
                    comp_data = next(iter(res["components"].values()))
                    record = {
                        "simulation_id": simulation_id,
                        "time": res["time"],
                        "Q_gas": comp_data.get("Q_gas"),
                        "Q_ch4": comp_data.get("Q_ch4"),
                        "Q_co2": comp_data.get("Q_co2"),
                        "pH": comp_data.get("pH"),
                        "VFA": comp_data.get("VFA"),
                        "TAC": comp_data.get("TAC"),
                    }
                    if record["Q_gas"] and record["Q_ch4"] and record["Q_gas"] > 0:
                        record["CH4_content"] = (record["Q_ch4"] / record["Q_gas"]) * 100
                    ts_records.append(record)
                session.bulk_insert_mappings(SimulationTimeSeries, ts_records)
                return sim
            except IntegrityError:
                raise ValueError(f"Simulation with ID '{simulation_id}' already exists")

    def load_simulation(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        with self.get_session() as session:
            sim = session.query(Simulation).filter(Simulation.id == simulation_id).first()
            if not sim:
                return None
            ts = (
                session.query(SimulationTimeSeries)
                .filter(SimulationTimeSeries.simulation_id == simulation_id)
                .order_by(SimulationTimeSeries.time)
                .all()
            )
            df = (
                pd.DataFrame(
                    {
                        c.name: [getattr(t, c.name) for t in ts]
                        for c in SimulationTimeSeries.__table__.columns
                        if c.name not in ["id", "simulation_id"]
                    }
                )
                if ts
                else pd.DataFrame()
            )
            return {**{c.name: getattr(sim, c.name) for c in Simulation.__table__.columns}, "time_series": df}

    def list_simulations(self, plant_id: Optional[str] = None, scenario: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.get_session() as session:
            query = session.query(Simulation)
            if plant_id:
                query = query.filter(Simulation.plant_id == plant_id)
            if scenario:
                query = query.filter(Simulation.scenario == scenario)
            simulations = query.order_by(Simulation.created_at.desc()).all()
            return [
                {
                    "id": s.id,
                    "plant_id": s.plant_id,
                    "name": s.name,
                    "scenario": s.scenario,
                    "duration": s.duration,
                    "avg_Q_ch4": s.avg_Q_ch4,
                    "status": s.status,
                    "created_at": s.created_at,
                }
                for s in simulations
            ]

    def store_calibration(
        self,
        plant_id: str,
        calibration_type: str,
        method: str,
        parameters: Dict[str, float],
        objective_value: float,
        objectives: List[str],
        validation_metrics: Optional[Dict[str, float]] = None,
        data_start: Optional[datetime] = None,
        data_end: Optional[datetime] = None,
        success: bool = True,
        message: Optional[str] = None,
    ) -> Calibration:
        with self.get_session() as session:
            cal = Calibration(
                plant_id=plant_id,
                calibration_type=calibration_type,
                method=method,
                parameters=parameters,
                objective_value=objective_value,
                objectives=objectives,
                validation_metrics=validation_metrics,
                data_start=data_start,
                data_end=data_end,
                success=success,
                message=message,
            )
            session.add(cal)
            return cal

    def load_calibrations(
        self, plant_id: str, calibration_type: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        with self.get_session() as session:
            query = session.query(Calibration).filter(Calibration.plant_id == plant_id)
            if calibration_type:
                query = query.filter(Calibration.calibration_type == calibration_type)
            cals = query.order_by(Calibration.created_at.desc()).limit(limit).all()
            return [{c.name: getattr(cal, c.name) for c in Calibration.__table__.columns} for cal in cals]

    def get_latest_calibration(self, plant_id: str) -> Optional[Dict[str, Any]]:
        cals = self.load_calibrations(plant_id, limit=1)
        return cals[0] if cals else None

    def store_substrate(
        self,
        plant_id: str,
        substrate_name: str,
        substrate_type: str,
        sample_date: Union[str, datetime],
        lab_data: Dict[str, float],
        sample_id: Optional[str] = None,
        lab_name: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Substrate:
        if isinstance(sample_date, str):
            sample_date = pd.to_datetime(sample_date)
        session = self.SessionLocal()
        try:
            substrate = Substrate(
                plant_id=plant_id,
                substrate_name=substrate_name,
                substrate_type=substrate_type,
                sample_date=sample_date,
                sample_id=sample_id,
                lab_name=lab_name,
                notes=notes,
            )
            for key, value in lab_data.items():
                if hasattr(substrate, key):
                    setattr(substrate, key, value)
            session.add(substrate)
            session.commit()
            session.refresh(substrate)
            session.expunge(substrate)
            return substrate
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def load_substrates(
        self,
        plant_id: str,
        substrate_type: Optional[str] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
    ) -> pd.DataFrame:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        with self.get_session() as session:
            query = session.query(Substrate).filter(Substrate.plant_id == plant_id)
            if substrate_type:
                query = query.filter(Substrate.substrate_type == substrate_type)
            if start_date:
                query = query.filter(Substrate.sample_date >= start_date)
            if end_date:
                query = query.filter(Substrate.sample_date <= end_date)
            substrates = query.order_by(Substrate.sample_date).all()
            if not substrates:
                return pd.DataFrame()
            cols = [
                "sample_date",
                "substrate_name",
                "substrate_type",
                "sample_id",
                "TS",
                "VS",
                "oTS",
                "foTS",
                "RP",
                "RL",
                "RF",
                "NDF",
                "ADF",
                "ADL",
                "pH",
                "NH4_N",
                "TAC",
                "COD_S",
                "BMP",
                "C_to_N",
                "lab_name",
            ]
            return pd.DataFrame([{c: getattr(s, c) for c in cols} for s in substrates])

    def _calculate_simulation_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {}
        q_gas, q_ch4, ph, vfa, p_el = [], [], [], [], []
        for res in results:
            comp_data = next(iter(res["components"].values()))
            if "Q_gas" in comp_data:
                q_gas.append(comp_data["Q_gas"])
            if "Q_ch4" in comp_data:
                q_ch4.append(comp_data["Q_ch4"])
            if "pH" in comp_data:
                ph.append(comp_data["pH"])
            if "VFA" in comp_data:
                vfa.append(comp_data["VFA"])
            for comp_res in res["components"].values():
                if "P_el" in comp_res:
                    p_el.append(comp_res["P_el"])
        metrics = {}
        if q_gas:
            metrics["avg_Q_gas"] = float(np.mean(q_gas))
        if q_ch4:
            metrics["avg_Q_ch4"] = float(np.mean(q_ch4))
            if q_gas:
                metrics["avg_CH4_content"] = float(np.mean(q_ch4) / np.mean(q_gas) * 100)
        if ph:
            metrics["avg_pH"] = float(np.mean(ph))
        if vfa:
            metrics["avg_VFA"] = float(np.mean(vfa))
        if p_el:
            metrics["total_energy"] = float(np.mean(p_el) * results[-1]["time"] * 24)
        return metrics

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        dangerous = ["DROP", "DELETE", "TRUNCATE", "ALTER"]
        if any(kw in query.upper() for kw in dangerous):
            raise ValueError("Dangerous keyword detected in query")
        return pd.read_sql(query, self.engine, params=params)

    def get_statistics(self, plant_id: str) -> Dict[str, Any]:
        with self.get_session() as session:
            return {
                "plant_id": plant_id,
                "n_measurements": session.query(Measurement).filter(Measurement.plant_id == plant_id).count(),
                "n_simulations": session.query(Simulation).filter(Simulation.plant_id == plant_id).count(),
                "n_calibrations": session.query(Calibration).filter(Calibration.plant_id == plant_id).count(),
                "n_substrates": session.query(Substrate).filter(Substrate.plant_id == plant_id).count(),
                "first_measurement": (
                    session.query(Measurement.timestamp)
                    .filter(Measurement.plant_id == plant_id)
                    .order_by(Measurement.timestamp)
                    .first()[0]
                    if session.query(Measurement.timestamp).filter(Measurement.plant_id == plant_id).count() > 0
                    else None
                ),
                "last_measurement": (
                    session.query(Measurement.timestamp)
                    .filter(Measurement.plant_id == plant_id)
                    .order_by(Measurement.timestamp.desc())
                    .first()[0]
                    if session.query(Measurement.timestamp).filter(Measurement.plant_id == plant_id).count() > 0
                    else None
                ),
            }

    def close(self) -> None:
        self.engine.dispose()
