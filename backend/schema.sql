-- PostgreSQL Schema for SatyaDrishti
-- To be executed in your Supabase SQL Editor

-- 1. Users Table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Analysis History Table
CREATE TABLE IF NOT EXISTS analysis_history (
    id VARCHAR(255) PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    user_name VARCHAR(255) DEFAULT 'Guest',
    file_name VARCHAR(255) NOT NULL,
    media_type VARCHAR(50) NOT NULL,
    prediction VARCHAR(50) DEFAULT 'PROCESSING',
    confidence DOUBLE PRECISION DEFAULT 0.0,
    report_path VARCHAR(500),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Agent Timeline Logs Table
CREATE TABLE IF NOT EXISTS agent_logs (
    id SERIAL PRIMARY KEY,
    analysis_id VARCHAR(255) REFERENCES analysis_history(id) ON DELETE CASCADE,
    agent_name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    duration DOUBLE PRECISION NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. Reports Table
CREATE TABLE IF NOT EXISTS reports (
    id SERIAL PRIMARY KEY,
    analysis_id VARCHAR(255) UNIQUE REFERENCES analysis_history(id) ON DELETE CASCADE,
    report_url VARCHAR(500) NOT NULL,
    file_size INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
