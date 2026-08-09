# CGD-QCSF Minimal Runnable Framework

This directory contains the minimal runnable core released with the
CGD-QCSF benchmark. It includes the directed state graph, execution state,
answer and execution contracts, evidence reviewer, SQL Code Generation Agent
(SCGA), Spatio-Temporal Computation Agent (STCA), schema retrieval, execution
adapters, and the principal prompt templates used in the experiments.

## Released Components

- `agent/`: orchestration graph, state manager, contracts, routing policy,
  evidence review, and bounded recovery.
- `prompts/`: intent-understanding, Text-to-SQL, and STCA code-generation
  prompt templates.
- `tools/`: schema retrieval, read-only SQL execution, Python subprocess
  execution, and map rendering.
- `schemas/`: semantic table descriptions used by schema pre-filtering.
- `utils/`: schema, code, and intermediate-artifact helpers.
- `code_templates/`: an empty dynamic-template registry. Experiment-generated
  scripts are intentionally not included.

## Setup

1. Restore the database snapshot from the repository root:

   ```powershell
   psql -U <user> -d <database> -f ..\create_database.sql
   ```

2. Install the framework dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and replace every placeholder with your own
   endpoint and database settings. Never commit `.env`.

4. Run one question from this directory:

   ```powershell
   python main.py -q "2015 年美国哪个州的年均 PM2.5 最高？"
   ```

The first schema-retrieval call may create a local Chroma index under
`workspace/chroma_db`. This directory is ignored by Git.

## Prompt Parity

The three files in `prompts/` are the shared principal templates used across
the compared foundation models. Model-specific handling is limited to
provider configuration and compatibility with tool-calling or structured-
output interfaces; the task instructions are not replaced by model-specific
question templates.

## Security Boundary

SQL execution is restricted to read-only statements and bounded result sizes.
Generated Python is checked and executed in a separate subprocess with a time
limit and restricted input staging. This local subprocess mechanism is an
experimental safety layer, not a production-grade container or operating-
system sandbox. Untrusted deployment should use stronger isolation such as a
container or dedicated sandbox service.

## Excluded Artifacts

The release excludes API keys, database passwords, local paths, raw model
traces, experiment logs, generated Chroma data, temporary workspaces, and
dynamically learned code templates.
