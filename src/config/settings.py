"""
Pipeline configuration dataclasses and settings management.

Supports loading from YAML files, environment variables, and
direct instantiation. Provides typed configuration for all
pipeline components including Cassandra, Spark, and models.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path

import yaml


@dataclass
class CassandraConfig:
    """
    Apache Cassandra / AWS Keyspaces connection configuration.

    Attributes:
        contact_points: List of Cassandra node addresses.
        port: Native transport port.
        keyspace: Target keyspace name.
        username: Authentication username.
        password: Authentication password.
        ssl_enabled: Enable SSL/TLS for AWS Keyspaces.
        ssl_certfile: Path to SSL certificate bundle.
        consistency_level: Default read/write consistency.
        default_ttl_seconds: Default TTL for inserted data.
        connection_timeout: Connection timeout in seconds.
        request_timeout: Per-query timeout in seconds.
        max_connections_per_host: Connection pool size per host.
        protocol_version: CQL native protocol version.
        retry_policy: Retry policy class name.
        dc_aware_routing: Enable DC-aware load balancing.
        local_dc: Local data center name for routing.
    """
    contact_points: List[str] = field(default_factory=lambda: ["127.0.0.1"])
    port: int = 9042
    keyspace: str = "timeseries"
    username: Optional[str] = None
    password: Optional[str] = None
    ssl_enabled: bool = False
    ssl_certfile: Optional[str] = None
    consistency_level: str = "LOCAL_QUORUM"
    default_ttl_seconds: int = 0
    connection_timeout: int = 10
    request_timeout: int = 30
    max_connections_per_host: int = 8
    protocol_version: int = 4
    retry_policy: str = "default"
    dc_aware_routing: bool = True
    local_dc: str = "datacenter1"

    def __post_init__(self):
        self.contact_points = (
            os.getenv("CASSANDRA_HOSTS", ",".join(self.contact_points))
            .split(",")
        )
        self.port = int(os.getenv("CASSANDRA_PORT", self.port))
        self.keyspace = os.getenv("CASSANDRA_KEYSPACE", self.keyspace)
        self.username = os.getenv("CASSANDRA_USERNAME", self.username)
        self.password = os.getenv("CASSANDRA_PASSWORD", self.password)
        self.ssl_enabled = os.getenv(
            "CASSANDRA_SSL", str(self.ssl_enabled)
        ).lower() == "true"


@dataclass
class SparkConfig:
    """
    Apache Spark session configuration.

    Attributes:
        app_name: Spark application name.
        master: Spark master URL.
        executor_memory: Memory allocation per executor.
        executor_cores: CPU cores per executor.
        num_executors: Number of executor instances.
        driver_memory: Driver process memory.
        shuffle_partitions: Number of shuffle partitions.
        default_parallelism: Default parallelism level.
        serializer: Serializer class.
        cassandra_connection_host: Spark-Cassandra connector host.
        cassandra_connection_port: Spark-Cassandra connector port.
        extra_config: Additional Spark configuration key-value pairs.
    """
    app_name: str = "TimeseriesMLPipeline"
    master: str = "local[*]"
    executor_memory: str = "4g"
    executor_cores: int = 4
    num_executors: int = 2
    driver_memory: str = "2g"
    shuffle_partitions: int = 200
    default_parallelism: int = 8
    serializer: str = "org.apache.spark.serializer.KryoSerializer"
    cassandra_connection_host: Optional[str] = None
    cassandra_connection_port: Optional[int] = None
    extra_config: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.master = os.getenv("SPARK_MASTER", self.master)
        self.executor_memory = os.getenv(
            "SPARK_EXECUTOR_MEMORY", self.executor_memory
        )


@dataclass
class ModelConfig:
    """
    Model training and hyperparameter configuration.

    Attributes:
        prophet_params: Prophet model hyperparameters.
        lightgbm_params: LightGBM model hyperparameters.
        cross_validation_folds: Number of time-series CV folds.
        train_test_split_ratio: Fraction of data for training.
        feature_lag_orders: Lag orders for feature generation.
        rolling_window_sizes: Window sizes for rolling statistics.
        fourier_order: Number of Fourier terms for seasonality.
        target_column: Name of the target variable column.
        date_column: Name of the datetime column.
        series_id_column: Column identifying individual series.
    """
    prophet_params: Dict[str, Any] = field(default_factory=lambda: {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "seasonality_mode": "multiplicative",
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "interval_width": 0.95,
    })
    lightgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
        "verbose": -1,
    })
    cross_validation_folds: int = 5
    train_test_split_ratio: float = 0.8
    feature_lag_orders: List[int] = field(
        default_factory=lambda: [1, 2, 3, 7, 14, 28]
    )
    rolling_window_sizes: List[int] = field(
        default_factory=lambda: [7, 14, 30, 90]
    )
    fourier_order: int = 5
    target_column: str = "value"
    date_column: str = "timestamp"
    series_id_column: str = "series_id"


@dataclass
class ForecastConfig:
    """
    Forecasting execution configuration.

    Attributes:
        horizon: Number of steps to forecast ahead.
        frequency: Time series frequency string (D, H, W, M, etc.).
        confidence_level: Prediction interval confidence level.
        ensemble_method: Ensemble combination strategy.
        ensemble_weights: Per-model weights for weighted ensemble.
        refit_frequency: How often to retrain models.
        min_training_points: Minimum data points required for training.
        max_training_points: Maximum data points used for training.
    """
    horizon: int = 30
    frequency: str = "D"
    confidence_level: float = 0.95
    ensemble_method: str = "weighted_average"
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "prophet": 0.4,
        "lightgbm": 0.4,
        "statsmodels": 0.2,
    })
    refit_frequency: str = "weekly"
    min_training_points: int = 60
    max_training_points: int = 10000


@dataclass
class ReconciliationConfig:
    """
    Hierarchical time series reconciliation configuration.

    Attributes:
        method: Reconciliation method (bottom_up, top_down, mint, ols).
        hierarchy_levels: Column names defining the hierarchy levels.
        top_down_method: Proportions method for top-down approach.
        mint_covariance: Covariance estimation for MinT method.
        non_negative: Enforce non-negative reconciled forecasts.
    """
    method: str = "mint"
    hierarchy_levels: List[str] = field(
        default_factory=lambda: ["total", "region", "store"]
    )
    top_down_method: str = "forecast_proportions"
    mint_covariance: str = "shrink"
    non_negative: bool = True


@dataclass
class PipelineSettings:
    """
    Top-level pipeline configuration aggregating all sub-configs.

    Provides factory methods for loading from YAML and environment.
    """
    cassandra: CassandraConfig = field(default_factory=CassandraConfig)
    spark: SparkConfig = field(default_factory=SparkConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    reconciliation: ReconciliationConfig = field(
        default_factory=ReconciliationConfig
    )

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineSettings":
        """
        Load pipeline settings from a YAML configuration file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Fully populated PipelineSettings instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the YAML is malformed.
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

        settings = cls()

        if "cassandra" in raw:
            settings.cassandra = CassandraConfig(**raw["cassandra"])
        if "spark" in raw:
            settings.spark = SparkConfig(**raw["spark"])
        if "model" in raw:
            settings.model = ModelConfig(**raw["model"])
        if "forecast" in raw:
            settings.forecast = ForecastConfig(**raw["forecast"])
        if "reconciliation" in raw:
            settings.reconciliation = ReconciliationConfig(
                **raw["reconciliation"]
            )

        return settings

    @classmethod
    def default(cls) -> "PipelineSettings":
        """Create settings with all default values."""
        return cls()
