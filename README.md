# NL2SQL Geospatial Benchmark (200 Questions)

This repository contains the public reproduction package for the benchmark used in the CGD-QCSF paper. It publishes the benchmark, database-construction material, strict evaluator, principal prompt templates, and a minimal runnable implementation of the core framework. It does not contain model API keys, database passwords, local paths, raw experiment logs, or generated runtime caches.

## Contents

| File | Description |
| --- | --- |
| `dev_plus.yaml` | 200 verified Chinese geospatial and spatio-temporal questions |
| `create_database.sql` | PostgreSQL/PostGIS schema and data snapshot used by the benchmark |
| `evaluate_experiments.py` | Strict structured evaluator used in the revised experiments |
| `framework/` | Minimal runnable CGD-QCSF core, prompt templates, schema descriptions, and execution adapters |
| `requirements.txt` | Minimal Python dependencies for reading the benchmark and running the evaluator |
| `LICENSE` | MIT License covering the released code |

## Benchmark Composition

| Split | Count |
| --- | ---: |
| U.S. state/spatial-unit questions | 100 |
| Zhejiang city/vector-fishnet questions | 100 |
| Easy | 66 |
| Medium | 64 |
| Hard | 70 |
| SQL-only reference path | 134 |
| SQL + Python reference path | 66 |
| **Total** | **200** |

The Zhejiang fishnet is stored as vector polygons with cell-level attributes. It is not a native raster dataset and should not be interpreted as PostGIS Raster or GeoTIFF pixel computation.

## YAML Fields

Each item contains:

| Field | Meaning |
| --- | --- |
| `question_id` | Stable integer ID |
| `domain` | `usa` or `zhejiang` |
| `question` | Chinese natural-language question |
| `difficulty` | `easy`, `medium`, or `hard` |
| `requires_sandbox` | Whether the verified reference path uses Python computation |
| `expected_sql_plan` | Reference SQL retrieval logic and output files |
| `expected_python_code` | Executable reference computation, or `null` for SQL-only tasks |
| `expected_execution_result` | Database/sandbox-verified structured result |

Reference SQL and Python describe one verified solution path. Evaluated systems may use different code as long as their final structured results are semantically equivalent.

## Database Setup

Create an empty PostgreSQL database with PostGIS available, then restore the supplied SQL file using a database account you control. Do not put credentials into this repository.

```powershell
psql -U <user> -d <database> -f create_database.sql
```

## Evaluator

Install the public evaluator dependencies:

```powershell
python -m pip install -r requirements.txt
```

Evaluate one or more run directories:

```powershell
python evaluate_experiments.py `
  --benchmark_path dev_plus.yaml `
  --run_dirs <run_dir_1> <run_dir_2> `
  --output_dir evaluation_results
```

Each run directory should contain one result YAML per question with the same fields produced by the released experiment format, including `metadata.question_id`, structured answer fields, execution metrics, token usage, and tool-routing trace.

The primary metric is `strict_structured_accuracy`. It checks answer-schema validity, entity–value pairing, list membership and order, numerical tolerance, units, exact integer identifiers, and tie-aware Top-K equivalence. `first_execution_path_success_rate` is auxiliary: it requires a correct strict answer with no failed tool call and no guardrail retry. A correct result obtained after bounded recovery remains correct under the primary metric.

## Core Framework and Prompt Templates

The released implementation is under [`framework/`](framework/). It contains the directed state graph, execution-state manager, SCGA and STCA adapters, answer and execution contracts, evidence review, bounded recovery, semantic schema retrieval, and the principal intent-understanding, Text-to-SQL, and Python code-generation prompts. See [`framework/README.md`](framework/README.md) for setup and execution instructions.

## Reproducibility Boundary

- Expected answers were obtained by running reference SQL and, when required, reference Python against the supplied database snapshot.
- The benchmark covers structured relational data, vector boundaries, and vector fishnet attributes.
- Native raster, trajectory streams, real-time sensors, open-web data discovery, ArcGIS Pro control, and QGIS desktop control are outside this release.
- Large model providers are external services; API availability, pricing, and model behavior may change.
- The local Python subprocess is an experimental execution boundary and should be replaced by stronger operating-system or container isolation for untrusted production deployment.

## Citation

Please cite the CGD-QCSF paper when using this benchmark. Formal citation metadata will be added after publication.
