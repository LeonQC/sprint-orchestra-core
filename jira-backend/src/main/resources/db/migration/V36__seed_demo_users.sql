-- The demo accounts used to be created by the frontend through public POST /api/users on an empty
-- database. That endpoint now needs the internal token, so a fresh database gets them here instead.
-- Password "123", stored as a bcrypt hash (see V35). Existing rows are left alone.
INSERT INTO users (username, name, email, password, avatar_color, created_at, updated_at) VALUES
    ('alice',   'Alice',   'alice@example.com',   '$2a$10$.7G43/qDfW6zKRPSphqVg.eF9SXvKTV9HLmlrSOsa48.W1sFXqYoi', 'linear-gradient(135deg, #6366f1, #7c3aed)', NOW(), NOW()),
    ('john',    'John',    'john@example.com',    '$2a$10$.7G43/qDfW6zKRPSphqVg.eF9SXvKTV9HLmlrSOsa48.W1sFXqYoi', 'linear-gradient(135deg, #06b6d4, #3b82f6)', NOW(), NOW()),
    ('charles', 'Charles', 'charles@example.com', '$2a$10$.7G43/qDfW6zKRPSphqVg.eF9SXvKTV9HLmlrSOsa48.W1sFXqYoi', 'linear-gradient(135deg, #10b981, #06b6d4)', NOW(), NOW())
ON CONFLICT DO NOTHING;
