"""
Cassandra time series storage client.

Provides a unified interface for persisting and querying time series
data in Apache Cassandra. Falls back to an in-memory pandas store
when Cassandra is not available, making local development and demos
possible without a running cluster.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from cassandra.cluster import Cluster
    from cassandra.auth import PlainTextAuthProvider
    from cassandra.policies import (
        DCAwareRoundRobinPolicy,
        RetryPolicy,
        ConstantReconnectionPolicy,
    )
    from cassandra.query import SimpleStatement, BatchStatement, BatchType

    CASSANDRA_AVAILABLE = True
except ImportError:
    CASSANDRA_AVAILABLE = False
    logger.info("cassandra-driver not installed; using in-memory simulation")


class CassandraClient:
    """
    Client for reading and writing time series data.

    When ``cassandra-driver`` is installed and a cluster is reachable the
    client talks to a real Cassandra backend.  Otherwise it transparently
    falls back to a local pandas DataFrame store so the rest of the
    pipeline can run without infrastructure dependencies.

    Parameters
    ----------
    contact_points : list[str]
        Cassandra node addresses.
    port : int
        Native CQL port (default 9042).
    keyspace : str
        Target keyspace name.
    username : str | None
        Authentication username.
    password : str | None
        Authentication password.
    local_dc : str
        Data center name for DC-aware routing.
    """

    TABLE_NAME = "timeseries"

    CREATE_TABLE_CQL = """
        CREATE TABLE IF NOT EXISTS {keyspace}.{table} (
            series_id   text,
            timestamp   timestamp,
            value       double,
            metadata    map<text, text>,
            PRIMARY KEY ((series_id), timestamp)
        ) WITH CLUSTERING ORDER BY (timestamp ASC)
          AND default_time_to_live = 0
          AND gc_grace_seconds = 864000;
    """

    INSERT_CQL = """
        INSERT INTO {keyspace}.{table} (series_id, timestamp, value, metadata)
        VALUES (?, ?, ?, ?)
    """

    SELECT_CQL = """
        SELECT series_id, timestamp, value, metadata
        FROM {keyspace}.{table}
        WHERE series_id = ?
        ORDER BY timestamp ASC
    """

    SELECT_RANGE_CQL = """
        SELECT series_id, timestamp, value, metadata
        FROM {keyspace}.{table}
        WHERE series_id = ?
          AND timestamp >= ?
          AND timestamp <= ?
        ORDER BY timestamp ASC
    """

    def __init__(
        self,
        contact_points: Optional[List[str]] = None,
        port: int = 9042,
        keyspace: str = "timeseries",
        username: Optional[str] = None,
        password: Optional[str] = None,
        local_dc: str = "datacenter1",
    ) -> None:
        self.contact_points = contact_points or ["127.0.0.1"]
        self.port = port
        self.keyspace = keyspace
        self.username = username
        self.password = password
        self.local_dc = local_dc

        self._cluster: Any = None
        self._session: Any = None
        self._prepared_insert: Any = None
        self._prepared_select: Any = None
        self._prepared_range: Any = None

        self._use_simulation = not CASSANDRA_AVAILABLE
        self._local_store: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Establish a connection to Cassandra or activate simulation."""
        if self._use_simulation:
            logger.info("Cassandra simulation mode active (in-memory store)")
            return

        try:
            auth = None
            if self.username and self.password:
                auth = PlainTextAuthProvider(
                    username=self.username,
                    password=self.password,
                )

            load_balancing = DCAwareRoundRobinPolicy(local_dc=self.local_dc)

            self._cluster = Cluster(
                contact_points=self.contact_points,
                port=self.port,
                auth_provider=auth,
                load_balancing_policy=load_balancing,
                reconnection_policy=ConstantReconnectionPolicy(delay=5.0),
                protocol_version=4,
            )
            self._session = self._cluster.connect()
            logger.info(
                "Connected to Cassandra at %s:%s",
                self.contact_points,
                self.port,
            )
        except Exception as exc:
            logger.warning(
                "Cannot reach Cassandra (%s); falling back to simulation", exc
            )
            self._use_simulation = True

    def disconnect(self) -> None:
        """Shut down the cluster connection."""
        if self._cluster is not None:
            self._cluster.shutdown()
            logger.info("Disconnected from Cassandra")
        self._cluster = None
        self._session = None

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def create_keyspace(self, replication_factor: int = 1) -> None:
        """Create the keyspace if it does not exist."""
        if self._use_simulation:
            logger.info("Simulation: keyspace '%s' ready", self.keyspace)
            return

        cql = (
            f"CREATE KEYSPACE IF NOT EXISTS {self.keyspace} "
            f"WITH replication = {{'class': 'SimpleStrategy', "
            f"'replication_factor': {replication_factor}}}"
        )
        self._session.execute(cql)
        self._session.set_keyspace(self.keyspace)
        logger.info("Keyspace '%s' created / verified", self.keyspace)

    def create_table(self) -> None:
        """Create the time series table using the canonical schema."""
        if self._use_simulation:
            logger.info("Simulation: table '%s' ready", self.TABLE_NAME)
            return

        cql = self.CREATE_TABLE_CQL.format(
            keyspace=self.keyspace, table=self.TABLE_NAME
        )
        self._session.execute(cql)
        self._prepare_statements()
        logger.info("Table '%s.%s' created / verified", self.keyspace, self.TABLE_NAME)

    def _prepare_statements(self) -> None:
        """Prepare CQL statements for efficient reuse."""
        self._prepared_insert = self._session.prepare(
            self.INSERT_CQL.format(keyspace=self.keyspace, table=self.TABLE_NAME)
        )
        self._prepared_select = self._session.prepare(
            self.SELECT_CQL.format(keyspace=self.keyspace, table=self.TABLE_NAME)
        )
        self._prepared_range = self._session.prepare(
            self.SELECT_RANGE_CQL.format(keyspace=self.keyspace, table=self.TABLE_NAME)
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def insert_timeseries(
        self,
        series_id: str,
        timestamp: datetime,
        value: float,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """Insert a single time series observation."""
        meta = metadata or {}

        if self._use_simulation:
            row = {
                "series_id": series_id,
                "timestamp": pd.Timestamp(timestamp),
                "value": float(value),
                "metadata": meta,
            }
            if series_id not in self._local_store:
                self._local_store[series_id] = pd.DataFrame(
                    columns=["series_id", "timestamp", "value", "metadata"]
                )
            new_row = pd.DataFrame([row])
            self._local_store[series_id] = pd.concat(
                [self._local_store[series_id], new_row], ignore_index=True
            )
            return

        self._session.execute(
            self._prepared_insert, (series_id, timestamp, value, meta)
        )

    def insert_dataframe(
        self,
        df: pd.DataFrame,
        series_id_col: str = "series_id",
        timestamp_col: str = "timestamp",
        value_col: str = "value",
        metadata_col: Optional[str] = "metadata",
        batch_size: int = 50,
    ) -> int:
        """
        Bulk-insert rows from a DataFrame.

        Returns the number of rows inserted.
        """
        count = 0

        if self._use_simulation:
            for _, row in df.iterrows():
                sid = str(row[series_id_col])
                ts = pd.Timestamp(row[timestamp_col])
                val = float(row[value_col])
                meta = row.get(metadata_col, {}) if metadata_col and metadata_col in df.columns else {}
                if not isinstance(meta, dict):
                    meta = {}
                self.insert_timeseries(sid, ts.to_pydatetime(), val, meta)
                count += 1
            logger.info("Simulation: inserted %d rows", count)
            return count

        batch = BatchStatement(batch_type=BatchType.UNLOGGED)
        for idx, row_data in df.iterrows():
            sid = str(row_data[series_id_col])
            ts = pd.Timestamp(row_data[timestamp_col]).to_pydatetime()
            val = float(row_data[value_col])
            meta = (
                row_data.get(metadata_col, {})
                if metadata_col and metadata_col in df.columns
                else {}
            )
            if not isinstance(meta, dict):
                meta = {}

            batch.add(self._prepared_insert, (sid, ts, val, meta))
            count += 1

            if count % batch_size == 0:
                self._session.execute(batch)
                batch = BatchStatement(batch_type=BatchType.UNLOGGED)

        if count % batch_size != 0:
            self._session.execute(batch)

        logger.info("Inserted %d rows into Cassandra", count)
        return count

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def query_timeseries(self, series_id: str) -> pd.DataFrame:
        """Retrieve all data for a given series as a DataFrame."""
        if self._use_simulation:
            if series_id in self._local_store:
                result = self._local_store[series_id].copy()
                result = result.sort_values("timestamp").reset_index(drop=True)
                return result
            return pd.DataFrame(
                columns=["series_id", "timestamp", "value", "metadata"]
            )

        rows = self._session.execute(self._prepared_select, (series_id,))
        records = [
            {
                "series_id": r.series_id,
                "timestamp": r.timestamp,
                "value": r.value,
                "metadata": r.metadata or {},
            }
            for r in rows
        ]
        return pd.DataFrame(records)

    def query_by_range(
        self,
        series_id: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Retrieve data for a series within a time range."""
        if self._use_simulation:
            df = self.query_timeseries(series_id)
            if df.empty:
                return df
            mask = (df["timestamp"] >= pd.Timestamp(start)) & (
                df["timestamp"] <= pd.Timestamp(end)
            )
            return df.loc[mask].reset_index(drop=True)

        rows = self._session.execute(
            self._prepared_range, (series_id, start, end)
        )
        records = [
            {
                "series_id": r.series_id,
                "timestamp": r.timestamp,
                "value": r.value,
                "metadata": r.metadata or {},
            }
            for r in rows
        ]
        return pd.DataFrame(records)

    def list_series(self) -> List[str]:
        """Return a list of all stored series IDs (simulation only)."""
        if self._use_simulation:
            return list(self._local_store.keys())
        cql = f"SELECT DISTINCT series_id FROM {self.keyspace}.{self.TABLE_NAME}"
        rows = self._session.execute(cql)
        return [r.series_id for r in rows]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def is_simulation(self) -> bool:
        """Whether the client is running in local simulation mode."""
        return self._use_simulation

    def row_count(self, series_id: str) -> int:
        """Return the number of stored observations for a series."""
        df = self.query_timeseries(series_id)
        return len(df)
