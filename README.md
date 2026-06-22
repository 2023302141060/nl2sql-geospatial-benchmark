# dev_plus.yaml Dataset Description

`dev_plus.yaml` is a Chinese NL2SQL / Agent benchmark dataset for geospatial and spatiotemporal data analysis tasks.

Each sample contains a Chinese natural language question, a reference SQL execution plan, optional Python analysis code, and expected results for evaluation. The dataset is suitable for evaluating model performance on natural language to SQL generation, spatial data querying, statistical analysis, and multi-step SQL + Python reasoning tasks.

## Dataset Size

The current file contains 80 benchmark samples:

| Category | Count |
|---|---:|
| U.S. state-level environmental and land-cover questions | 40 |
| Zhejiang city and fishnet-grid questions | 40 |
| easy | 26 |
| medium | 24 |
| hard | 30 |
| SQL-only tasks | 54 |
| SQL + Python analysis tasks | 26 |

## File Format

`dev_plus.yaml` is encoded in UTF-8. Its top-level structure is a YAML list, where each item represents one benchmark question. The main fields are:

| Field | Description |
|---|---|
| `question_id` | Unique question ID |
| `domain` | Data domain, currently `usa` or `zhejiang` |
| `question` | Chinese natural language question |
| `difficulty` | Difficulty label: `easy`, `medium`, or `hard` |
| `requires_sandbox` | Whether additional Python sandbox analysis is required |
| `expected_sql_plan` | Reference SQL execution plan containing one or more SQL queries |
| `expected_sql_plan.queries[].sql` | Reference SQL statement |
| `expected_sql_plan.queries[].output_filename` | Suggested filename for saving the SQL result |
| `expected_sql_plan.queries[].has_geometry` | Whether the query result contains geometry fields |
| `expected_python_code` | Reference Python code for secondary analysis; `null` for SQL-only tasks |
| `expected_execution_result` | Expected result or key result subset for evaluating correctness |

## Related Files

The benchmark can be shared together with the following database setup files:

| File | Description |
|---|---|
| `benchmark/dev_plus.yaml` | Main benchmark dataset |
| `benchmark/README.md` | Dataset description |
| `create_database.sql` | PostgreSQL/PostGIS SQL dump for creating and populating the evaluation database |
| `reqiurements.txt` | Python environment dependency list for database setup and reproduction |

## Example Use Cases

This dataset can be used to:

- test whether an NL2SQL model can generate correct SQL;
- test whether an Agent can complete SQL queries and Python analysis step by step;
- compare different models or Agent frameworks on geospatial and spatiotemporal analysis tasks;
- build an automated benchmark using `question_id`, `expected_sql_plan`, and `expected_execution_result` for result validation.

## Notes

- `dev_plus.yaml` contains benchmark questions, reference solutions, and expected results.
- `create_database.sql` provides the database schema and data needed to reproduce the benchmark environment.
- The SQL table names and column names must match the target evaluation database.
- For GitHub sharing, upload `dev_plus.yaml`, this README, `create_database.sql`, and `reqiurements.txt` while preserving the relative paths shown above.
