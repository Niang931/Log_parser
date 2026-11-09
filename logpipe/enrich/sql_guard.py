
"""
SQL security policy for Logpipe.

All SQL must pass through validate_sql() before being returned as an
approved query or sent to a database executor.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

MAX_SQL_LENGTH = 20_000

ALLOWED_TABLES = {
    "sqlite": {"logs"},
    "clickhouse": {"iot_logs"},
}

# Only columns belonging to the application's log tables.
# Update these if the actual database schemas change.
ALLOWED_COLUMNS = {
    "sqlite": {
        "log_id",
        "timestamp",
        "device_id",
        "service",
        "source",
        "template_id",
        "status",
        "payload",
        "raw_message",
        "level",
    },
    "clickhouse": {
        "log_id",
        "timestamp",
        "device_id",
        "service",
        "source",
        "template_id",
        "log_status",
        "payload",
        "raw_message",
        "tags",
    },
}

# Functions that can access external resources, modify state,
# reveal privileged metadata, or cause deliberate delays.
#
# This list is defense-in-depth. Table and column authorization
# and database read-only permissions are still required.
BLOCKED_FUNCTIONS = {
    "sleep",
    "pg_sleep",
    "benchmark",
    "load_extension",
    "readfile",
    "writefile",
    "read_csv",
    "read_csv_auto",
    "read_json",
    "read_json_auto",
    "read_parquet",
    "read_blob",
    "url",
    "remote",
    "remotesecure",
    "cluster",
    "clusterallreplicas",
    "file",
    "s3",
    "hdfs",
    "azureblobstorage",
    "mysql",
    "postgresql",
    "jdbc",
    "odbc",
    "executable",
    "input",
    "generaterandom",
    "current_user",
    "current_role",
    "current_database",
    "current_catalog",
    "current_schema",
    "version",
}

# SQLGlot expression types that must never appear in an approved
# query, including when nested inside another expression.
BLOCKED_NODE_NAMES = {
    "Insert",
    "Update",
    "Delete",
    "Drop",
    "Create",
    "Alter",
    "TruncateTable",
    "Merge",
    "Grant",
    "Revoke",
    "Command",
    "Transaction",
    "Commit",
    "Rollback",
    "Pragma",
    "Attach",
    "Detach",
    "Use",
    "Set",
    "Into",
    "Lock",
}


def _reject(message: str) -> None:
    raise ValueError(f"SQL rejected: {message}")


def _node_name(node: exp.Expression) -> str:
    return type(node).__name__


def _function_name(node: exp.Expression) -> str:
    """Get a normalized SQL function name."""
    if isinstance(node, exp.Anonymous):
        return node.name.lower()
    return str(node.sql_name()).lower()


def _get_cte_names(tree: exp.Expression) -> set[str]:
    """Collect CTE names so references to CTEs aren't mistaken for tables."""
    names = set()

    for cte in tree.find_all(exp.CTE):
        alias = cte.alias
        if alias:
            names.add(alias.lower())

    return names


def _get_table_aliases(tree: exp.Expression) -> set[str]:
    aliases = set()

    for table in tree.find_all(exp.Table):
        if table.alias:
            aliases.add(table.alias.lower())

        if table.name:
            aliases.add(table.name.lower())

    return aliases


def _contains_sql_comment(sql: str) -> bool:
    i = 0
    quote = None

    while i < len(sql):
        char = sql[i]

        if quote:
            if char == quote:
                # SQL escapes quote characters by doubling them.
                if i + 1 < len(sql) and sql[i + 1] == quote:
                    i += 2
                    continue
                quote = None

            elif char == "\\" and quote == "'" and i + 1 < len(sql):
                i += 2
                continue

            i += 1
            continue

        if char in ("'", '"', "`"):
            quote = char
            i += 1
            continue

        if char == "[":
            quote = "]"
            i += 1
            continue

        if sql.startswith("--", i):
            return True

        if sql.startswith("/*", i) or sql.startswith("*/", i):
            return True

        i += 1

    return False


def _check_tables(
    tree: exp.Expression,
    engine: str,
) -> None:
    allowed = ALLOWED_TABLES[engine]
    ctes = _get_cte_names(tree)

    for table in tree.find_all(exp.Table):
        # Reject table functions, including external data sources.
        if isinstance(table.this, exp.Func):
            _reject("table functions are not permitted")

        name = table.name.lower()

        # Reject database and catalog qualification.
        if table.db or table.catalog:
            _reject("database-qualified table references are not permitted")

        # CTE references are permitted; physical tables must be allowlisted.
        if name in ctes:
            continue

        if name not in allowed:
            _reject(f"table {name!r} is not authorized")


def _is_tautology(node: exp.Expression) -> bool:
    if isinstance(node, exp.Boolean):
        return node.this is True

    if isinstance(node, exp.EQ):
        left = node.this
        right = node.expression

        if (
            isinstance(left, exp.Literal)
            and isinstance(right, exp.Literal)
            and left.this == right.this
            and left.is_string == right.is_string
        ):
            return True

    return False


def _check_tautologies(tree: exp.Expression) -> None:
    for node in tree.find_all(exp.Or):
        if (
            _is_tautology(node.this)
            or _is_tautology(node.expression)
        ):
            _reject("always-true OR conditions are not permitted")


def _check_columns(
    tree: exp.Expression,
    engine: str,
) -> None:
    allowed = ALLOWED_COLUMNS[engine]
    aliases = _get_table_aliases(tree)

    # Aliases can legitimately be referenced in ORDER BY or GROUP BY.
    selected_aliases = {
        node.alias.lower()
        for node in tree.find_all(exp.Alias)
        if node.alias
    }

    # CTE output aliases can also be referenced by the outer query.
    cte_columns = set()

    for cte in tree.find_all(exp.CTE):
        for column in cte.find_all(exp.Column):
            cte_columns.add(column.name.lower())

        for identifier in cte.find_all(exp.Identifier):
            # Conservative support for explicitly named CTE columns.
            cte_columns.add(identifier.name.lower())

    allowed_names = allowed | selected_aliases | cte_columns

    for column in tree.find_all(exp.Column):
        name = column.name.lower()

        # SELECT * is allowed only because its source tables are
        # independently restricted to the application's log table.
        if isinstance(column.this, exp.Star):
            continue

        if name not in allowed_names:
            _reject(f"column {name!r} is not authorized")

        qualifier = column.table.lower()
        if qualifier and qualifier not in aliases:
            _reject(f"column qualifier {qualifier!r} is not authorized")


def _check_functions(tree: exp.Expression) -> None:
    for node in tree.walk():
        if not isinstance(node, exp.Func):
            continue

        name = _function_name(node)

        if name in BLOCKED_FUNCTIONS:
            _reject(f"function {name!r} is not permitted")


def _check_nodes(tree: exp.Expression) -> None:
    for node in tree.walk():
        if _node_name(node) in BLOCKED_NODE_NAMES:
            _reject(f"SQL operation {_node_name(node)!r} is not permitted")


def validate_sql(sql: str, engine: str) -> None:
    if engine not in ALLOWED_TABLES:
        _reject(f"unsupported database engine {engine!r}")

    if not isinstance(sql, str):
        _reject("SQL must be a string")

    sql = sql.strip().rstrip(";")

    if not sql:
        _reject("SQL is empty")

    if _contains_sql_comment(sql):
        _reject("SQL comments are not permitted")

    if len(sql) > MAX_SQL_LENGTH:
        _reject("SQL exceeds the maximum permitted length")

    try:
        statements = sqlglot.parse(
            sql,
            read=engine,
        )
    except (SqlglotError, RecursionError) as error:  # deep nesting overflows sqlglot's parser
        _reject(f"SQL could not be parsed ({str(error).splitlines()[0]})")

    # Reject empty input and multiple statements.
    statements = [statement for statement in statements if statement is not None]

    if len(statements) != 1:
        _reject("exactly one SQL statement is required")

    tree = statements[0]

    # Only SELECT-style query roots are accepted.
    #
    # UNION, INTERSECT, and EXCEPT are query expressions too.
    allowed_roots = (
        exp.Select,
        exp.Union,
        exp.Intersect,
        exp.Except,
    )

    if not isinstance(tree, allowed_roots):
        _reject("only SELECT queries and SELECT-based CTEs are permitted")

    # Check the entire tree, not just its first keyword.
    _check_nodes(tree)
    _check_tables(tree, engine)
    _check_columns(tree, engine)
    _check_functions(tree)
    _check_tautologies(tree)


if __name__ == "__main__":
    """test my sql guard here. move to /tests or delete if want"""

    # Valid queries
    VALID = [
        ("SELECT 1", "sqlite"),
        ("SELECT timestamp, service FROM logs LIMIT 10", "sqlite"),
        (
            "WITH sample AS (SELECT 1 AS n) SELECT n FROM sample",
            "sqlite",
        ),
        ("SELECT 1 UNION SELECT 2", "sqlite"),
        ("SELECT timestamp, service FROM iot_logs LIMIT 10", "clickhouse"),
    ]

    # Invalid queries
    INVALID = [
        ("DROP TABLE logs", "sqlite"),
        ("DELETE FROM logs", "sqlite"),
        ("SELECT 1; DROP TABLE logs", "sqlite"),
        ("SELECT * FROM information_schema.tables", "sqlite"),
        ("SELECT name FROM sqlite_master", "sqlite"),
        ("SELECT * FROM system.tables", "clickhouse"),
        ("SELECT SLEEP(1)", "sqlite"),
        ("SELECT * FROM remote('host', 'db', 'table')", "clickhouse"),
    ]

    UNSAFE_FUNCTION_SQL = [
        "SELECT SLEEP(1)",
        "SELECT sleep(1)",
        "SELECT pg_sleep(1)",
        "SELECT BENCHMARK(1000000, MD5('test'))",
        "SELECT load_extension('test')",
        "SELECT readfile('/etc/passwd')",
        "SELECT writefile('/tmp/test', 'marker')",
    ]

    for sql, engine in VALID:
        validate_sql(sql, engine)
        print("PASS valid:", sql)

    for sql, engine in INVALID:
        try:
            validate_sql(sql, engine)
        except ValueError:
            print("PASS rejected:", sql)
        else:
            raise AssertionError(f"Unsafe SQL was accepted: {sql}")

    for sql in UNSAFE_FUNCTION_SQL:
        try:
            validate_sql(sql, "sqlite")
        except ValueError:
            print("PASS rejected:", sql)
        else:
            raise AssertionError(f"Unsafe SQL was accepted: {sql}")
