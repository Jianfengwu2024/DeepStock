DO $$
DECLARE
    item RECORD;
BEGIN
    FOR item IN
        SELECT constraint_name.conname
        FROM pg_constraint AS constraint_name
        JOIN pg_class AS table_name ON table_name.oid = constraint_name.conrelid
        JOIN pg_namespace AS schema_name ON schema_name.oid = table_name.relnamespace
        WHERE schema_name.nspname = 'bnhmm'
          AND table_name.relname = 'predictions'
          AND constraint_name.contype = 'c'
          AND pg_get_constraintdef(constraint_name.oid) LIKE '%array_length(posterior_mean%'
    LOOP
        EXECUTE format(
            'ALTER TABLE bnhmm.predictions DROP CONSTRAINT %I', item.conname
        );
    END LOOP;

    FOR item IN
        SELECT constraint_name.conname
        FROM pg_constraint AS constraint_name
        JOIN pg_class AS table_name ON table_name.oid = constraint_name.conrelid
        JOIN pg_namespace AS schema_name ON schema_name.oid = table_name.relnamespace
        WHERE schema_name.nspname = 'bnhmm'
          AND table_name.relname = 'predictions'
          AND constraint_name.contype = 'c'
          AND pg_get_constraintdef(constraint_name.oid) LIKE '%array_length(posterior_variance%'
    LOOP
        EXECUTE format(
            'ALTER TABLE bnhmm.predictions DROP CONSTRAINT %I', item.conname
        );
    END LOOP;
END $$;

ALTER TABLE bnhmm.predictions
    ADD CONSTRAINT predictions_posterior_mean_length
    CHECK (array_length(posterior_mean, 1) IN (4, 5)),
    ADD CONSTRAINT predictions_posterior_variance_length
    CHECK (array_length(posterior_variance, 1) IN (4, 5));

COMMENT ON CONSTRAINT predictions_posterior_mean_length ON bnhmm.predictions IS
    'New directional-four-v1 models write four values; five remains readable for historical versions.';
