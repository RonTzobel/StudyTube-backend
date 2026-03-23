-- Migration: add s3_key and original_filename to videos table
-- Run this once against your database before restarting the backend.
--
-- These columns are nullable so existing rows are unaffected.
-- Videos uploaded before this migration keep their file_path value and
-- will be handled by the legacy code path in the worker.

ALTER TABLE videos
    ADD COLUMN IF NOT EXISTS s3_key TEXT,
    ADD COLUMN IF NOT EXISTS original_filename TEXT;
