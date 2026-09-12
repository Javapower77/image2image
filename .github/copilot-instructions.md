# Local Photo Edit Studio instructions

- Use Python 3.11 and type annotations.
- Keep inference local; never add cloud inference, telemetry, or prompt/image logging.
- Keep model adapters lazy-loaded and preserve the one-model-at-a-time memory policy.
- Do not disable model-native safety mechanisms or license-required safeguards.
- Run `ruff check .` and `pytest` after changes.
- Update `CHANGELOG.md` for user-visible changes.
