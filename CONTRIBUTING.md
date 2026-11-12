# Contributing

Issues and pull requests are welcome, in English or Spanish.

## Development setup

```bash
git clone https://github.com/sergioide007/synthetic-code.git && cd synthetic-code
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,api]" ruff
pytest -q                                               # Redis/Postgres/Kafka tests skip unless their env vars are set
ruff check src tests demos examples scripts
```

`sh scripts/verify.sh` runs the full verification (tests, independent `jq` + `openssl` checks, ruff, mypy, bandit,
dependency audit), the same as the `quality` job in CI.

## Rules for a change to be merged

1. **Every behaviour change comes with a test**, and a fix comes with a test that fails without it.
2. **Examples stay offline.** Everything in `examples/` must run with no API key and end by printing `OK`
   (`tests/test_examples.py` enforces it).
3. **The core stays dependency-free.** Optional libraries go behind an extra in `pyproject.toml` and are imported
   lazily, so `import sintetico` keeps working without them.
4. **Fail closed.** A guard, gate or validator that crashes or lacks data must deny or escalate, never allow.
5. **Rules that protect are code, not prompts.** An LLM may explain a decision; it never overrides a `BLOCK`.
6. Write code, docstrings and commit messages in English. Use [Conventional Commits](https://www.conventionalcommits.org/)
   (`feat:`, `fix:`, `docs:`, `test:`, `ci:`).
7. Add a line to `CHANGELOG.md` under a new version heading.

## Releases

Versions follow [Semantic Versioning](https://semver.org/). To release: update the version in `pyproject.toml`,
`src/sintetico/__init__.py` and `CITATION.cff`, move the changelog entry under the new version, then create an
annotated tag `vX.Y.Z` on `main` once CI is green.

## Security

Do not report vulnerabilities in public issues: see [SECURITY.md](SECURITY.md).
