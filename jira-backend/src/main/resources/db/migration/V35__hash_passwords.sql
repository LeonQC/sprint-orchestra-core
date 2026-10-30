-- Passwords were stored and compared as plain text. Hash the existing ones with bcrypt; the app now
-- only writes and checks hashes. pgcrypto's bf output ($2a$) is what Spring's BCryptPasswordEncoder reads.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

UPDATE users
SET password = crypt(password, gen_salt('bf', 10))
WHERE password IS NOT NULL
  AND password NOT LIKE '$2_$%';
