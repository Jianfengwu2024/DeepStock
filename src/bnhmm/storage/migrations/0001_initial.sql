CREATE SCHEMA IF NOT EXISTS bnhmm;

CREATE TABLE IF NOT EXISTS bnhmm.model_versions (
    model_version TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'training'
        CHECK (status IN ('training', 'validated', 'published', 'rejected', 'retired')),
    architecture_version TEXT NOT NULL,
    code_commit TEXT NOT NULL,
    config JSONB NOT NULL,
    data_fingerprint TEXT NOT NULL,
    training_started_at TIMESTAMPTZ,
    training_finished_at TIMESTAMPTZ,
    training_cutoff TIMESTAMPTZ NOT NULL,
    published_at TIMESTAMPTZ,
    is_default BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (status <> 'published' OR published_at IS NOT NULL),
    CHECK (NOT is_default OR status = 'published')
);

CREATE UNIQUE INDEX IF NOT EXISTS model_versions_one_default_architecture
    ON bnhmm.model_versions (architecture_version)
    WHERE is_default;

CREATE TABLE IF NOT EXISTS bnhmm.model_artifacts (
    artifact_id BIGSERIAL PRIMARY KEY,
    model_version TEXT NOT NULL REFERENCES bnhmm.model_versions(model_version) ON DELETE CASCADE,
    artifact_kind TEXT NOT NULL,
    shard_key TEXT NOT NULL DEFAULT '',
    compression TEXT NOT NULL DEFAULT 'none',
    sha256 TEXT NOT NULL,
    byte_size BIGINT NOT NULL CHECK (byte_size >= 0),
    payload BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (model_version, artifact_kind, shard_key)
);

CREATE TABLE IF NOT EXISTS bnhmm.model_symbols (
    model_version TEXT NOT NULL REFERENCES bnhmm.model_versions(model_version) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    adapter_artifact_id BIGINT REFERENCES bnhmm.model_artifacts(artifact_id),
    valid_from TIMESTAMPTZ NOT NULL,
    valid_to TIMESTAMPTZ,
    training_observations BIGINT NOT NULL DEFAULT 0,
    enabled BOOLEAN NOT NULL DEFAULT true,
    PRIMARY KEY (model_version, symbol),
    CHECK (valid_to IS NULL OR valid_to > valid_from)
);

CREATE TABLE IF NOT EXISTS bnhmm.predictions (
    model_version TEXT NOT NULL REFERENCES bnhmm.model_versions(model_version) ON DELETE RESTRICT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL CHECK (timeframe IN ('60m', '1d')),
    signal_time TIMESTAMPTZ NOT NULL,
    available_at TIMESTAMPTZ NOT NULL,
    training_cutoff TIMESTAMPTZ NOT NULL,
    posterior_mean DOUBLE PRECISION[] NOT NULL,
    posterior_variance DOUBLE PRECISION[] NOT NULL,
    latent_probabilities DOUBLE PRECISION[] NOT NULL,
    aleatoric_uncertainty DOUBLE PRECISION NOT NULL CHECK (aleatoric_uncertainty >= 0),
    epistemic_uncertainty DOUBLE PRECISION NOT NULL CHECK (epistemic_uncertainty >= 0),
    expected_score DOUBLE PRECISION NOT NULL CHECK (expected_score BETWEEN -2 AND 2),
    default_target_weight DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (model_version, symbol, timeframe, signal_time),
    CHECK (available_at > signal_time),
    CHECK (training_cutoff <= signal_time),
    CHECK (array_length(posterior_mean, 1) = 5),
    CHECK (array_length(posterior_variance, 1) = 5),
    CHECK (array_length(latent_probabilities, 1) = 10)
) PARTITION BY RANGE (signal_time);

CREATE TABLE IF NOT EXISTS bnhmm.predictions_default
    PARTITION OF bnhmm.predictions DEFAULT;

CREATE INDEX IF NOT EXISTS predictions_lookup
    ON bnhmm.predictions (symbol, timeframe, signal_time, model_version);

CREATE TABLE IF NOT EXISTS bnhmm.training_jobs (
    job_id UUID PRIMARY KEY,
    model_version TEXT REFERENCES bnhmm.model_versions(model_version),
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    requested_config JSONB NOT NULL,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    heartbeat_at TIMESTAMPTZ,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS bnhmm.evaluation_metrics (
    model_version TEXT NOT NULL REFERENCES bnhmm.model_versions(model_version) ON DELETE CASCADE,
    scope TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    sample_count BIGINT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (model_version, scope, metric_name)
);

CREATE TABLE IF NOT EXISTS bnhmm.report_artifacts (
    report_id BIGSERIAL PRIMARY KEY,
    model_version TEXT NOT NULL REFERENCES bnhmm.model_versions(model_version) ON DELETE CASCADE,
    report_kind TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    payload BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (model_version, report_kind)
);
