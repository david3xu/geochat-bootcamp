-- Module 1: Data Foundation - PostGIS Validation Queries
-- For database initialization and validation

-- Enable PostGIS extensions if not already enabled
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Verify spatial reference systems
SELECT count(*) FROM spatial_ref_sys WHERE srid = 4326;

-- Create test point geometry
CREATE OR REPLACE FUNCTION validate_installation() RETURNS TEXT AS $$
DECLARE
    result TEXT;
BEGIN
    -- Create a simple point for testing
    WITH test_point AS (
        SELECT ST_SetSRID(ST_MakePoint(115.0, -32.0), 4326) AS geom
    )
    -- Test basic PostGIS functionality
    SELECT 
        CASE 
            WHEN ST_IsValid(geom) = true THEN 'PostGIS validation passed'
            ELSE 'PostGIS validation failed'
        END INTO result
    FROM test_point;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Run validation function
SELECT validate_installation();

-- Sample spatial query to test performance
CREATE OR REPLACE FUNCTION test_spatial_query_performance() RETURNS TABLE(
    query_name TEXT,
    execution_time_ms FLOAT
) AS $$
DECLARE
    start_time TIMESTAMPTZ;
    end_time TIMESTAMPTZ;
BEGIN
    -- Test 1: Point in polygon query
    query_name := 'Point in polygon query';
    start_time := clock_timestamp();
    
    PERFORM ST_Contains(
        ST_MakeEnvelope(112.0, -36.0, 130.0, -13.0, 4326),
        ST_SetSRID(ST_MakePoint(115.0, -32.0), 4326)
    );
    
    end_time := clock_timestamp();
    execution_time_ms := extract(epoch from (end_time - start_time)) * 1000;
    RETURN NEXT;
    
    -- Test 2: Distance calculation
    query_name := 'Distance calculation';
    start_time := clock_timestamp();
    
    PERFORM ST_Distance(
        ST_SetSRID(ST_MakePoint(115.0, -32.0), 4326)::geography,
        ST_SetSRID(ST_MakePoint(116.0, -31.0), 4326)::geography
    );
    
    end_time := clock_timestamp();
    execution_time_ms := extract(epoch from (end_time - start_time)) * 1000;
    RETURN NEXT;
    
    RETURN;
END;
$$ LANGUAGE plpgsql;

-- Optional: Execute performance test
-- SELECT * FROM test_spatial_query_performance();
