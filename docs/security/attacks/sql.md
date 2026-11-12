# SQL Injection

SQL injection occurs when untrusted data changes the meaning of a SQL statement.

In an LLM-assisted application, a log entry may manipulate the model into proposing unsafe SQL, or user input may be concatenated into a query.

## SQL obfuscation

Obfuscation attempts to bypass simplistic keyword filters by changing letter case, inserting comments, or varying whitespace.

SQL keywords are generally case-insensitive.

```sql
SeLeCt 1;
DrOp TABLE test_table;
SELECT/**/1
SELECT/* harmless test comment */1
SELECT
1
SELECT	1
SELECT
/* comment */
1
-- trailing comment
# MySQL-family line comment
/* block comment */
InTo OuTFILe
```

Also test mixed-case forms of sensitive keywords, such as:

```sql
AlTeR SyStEm;
AlTeR UsEr;
GrAnT;
ReVoKe;
```

## SQL injection through generated SQL

This occurs when untrusted text influences a generated SQL string, for example when an LLM copies attacker-controlled log content into a query.

The purpose may be to change query conditions or append an unauthorized operation.

```sql
'; DROP TABLE iot_logs; --

' OR '1'='1

' UNION SELECT NULL --
```

## SQL statement chaining

An attacker attempts to append a second statement to a legitimate query.

```sql
SELECT 1; DROP TABLE logs;
SELECT 1; DELETE FROM test_table;
```

The application should parse the complete request and permit exactly one authorized statement.

## File output and server-side file access

Some database engines provide file-output or file-reading features. If enabled and sufficiently privileged, these capabilities may expose server files or write files accessible to the database process.

A harmless MySQL-family file-output test:

```sql
SELECT 'LOGPIPE_SECURITY_TEST_MARKER'
INTO OUTFILE '/tmp/logpipe-security-test.txt';
```

Also test whether the policy rejects:

```sql
INTO DUMPFILE
```

Do not use web-shell content or a web-accessible destination in routine regression tests. A harmless marker in a disposable directory is sufficient.

## Administrative and privilege changing actions

Administrative statements can alter server configuration, user accounts, or privileges.

```sql
ALTER USER test_user IDENTIFIED BY 'test-only-password';
ALTER SYSTEM SET some_setting = 'test';
GRANT SELECT ON test_db.* TO test_user;
REVOKE SELECT ON test_db.* FROM test_user;
```

The read-only query path should reject administrative and privilege-management operations regardless of keyword casing.

## Destructive data operations

These statements can delete or replace application data.

```sql
DROP TABLE test_table;
DROP DATABASE test_db;
TRUNCATE TABLE test_table;
DELETE FROM test_table;
UPDATE test_table SET value = 'changed';
INSERT INTO test_table (value) VALUES ('injected');
```

## Data exfiltration and metadata discovery

A query can be read-only and still disclose information the caller is not authorized to see.

```sql
SELECT * FROM sensitive_table;
SELECT * FROM information_schema.tables;
SELECT * FROM system.tables;
SELECT 1 UNION SELECT 2;
' AND EXISTS (SELECT 1) --
' AND NOT EXISTS (SELECT 1 WHERE 1=1) --
' OR (SELECT 1)=1 --
```

Test access to application tables, database catalogs, system tables, and sensitive columns.

`UNION` is not inherently malicious; enforce authorization on the resulting query rather than banning it indiscriminately.

## CTEs and nested queries

A common query prefix does not prove that the entire statement is safe.

```sql
WITH sample AS (SELECT 1)
SELECT * FROM sample;
WITH sample AS (SELECT 1 AS n)
SELECT n FROM sample;
SELECT *
FROM (SELECT 1 AS n) AS sample;
```

## Boolean-condition manipulation

```sql
' OR 'a'='a
' OR 1=1 --
' AND 1=2 --
') OR ('a'='a
')) OR (1=1) --
' AND 'a'='b
1 OR 1=1
1 AND 1=2
0) OR (1=1
```

## Quote and delimiter

```sql
'
''
'''
"
""
\\
') 
'))
```

## Parentheses and nested expressions

```sql
' AND (1=1) --
' AND (1=2) --
') AND (1=1) --
' OR ((1=1) AND (2=2)) --
```

## Read-only policy and statement classification

```sql
EXPLAIN SELECT 1;
SHOW TABLES;
DESCRIBE test_table;
PRAGMA table_info(test_table);
```

## Metadata discovery

```sql
SELECT table_name FROM information_schema.tables;
SELECT column_name FROM information_schema.columns;
SELECT name FROM sqlite_master;
SELECT current_user;
SELECT current_database();
```

## Blind SQL injection

Use only in isolated test environments with strict timeouts.

```sql
-- MySQL-family
SELECT SLEEP(1);

-- PostgreSQL
SELECT pg_sleep(1);

-- SQL Server
WAITFOR DELAY '00:00:01';
```

## Error-based SQL injection

```sql
'
)
' AND (
SELECT
```

## Encoded and normalized input

```text
%27
%2527
%2D%2D
%2F%2A
%00
\u0027
\u002D\u002D
```

## Second-order SQL injection

Example synthetic stored value:

```text
test' OR '1'='1 -- marker
```

Test features that later retrieve, search, sort, export, or report on stored values.

## Identifier and sort-direction injection

```text
name
created_at
name DESC
name, (SELECT 1)
test_table
information_schema.tables
```

Use fixed allowlists for column names, table names, and sort directions.

## JSON and full-text query boundaries

```text
"
'
\\
$
.
*
(
)
OR
AND
```

## Security implementation recommendations

- Use prepared statements and parameterized queries.
- Avoid SQL string concatenation and interpolation.
- Allowlist identifiers and sort directions.
- Parse and validate the complete SQL statement.
- Enforce authorization on tables, columns, functions, and returned data.
- Use least-privilege database accounts.
- Disable multiple-statement execution unless required.
- Treat LLM output and log contents as untrusted input.

## Source
https://owasp.org/www-community/attacks/SQL_Injection
https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html
