# Testing

Run the full test suite:

```bash
pip install -e .[test]
pytest tests/ -v
```

Test philosophies and notes
- Tests are written to skip gracefully when optional dependencies are missing.
- Add a unit test for any new model, transform, or persistence routine.

Where to add tests
- `tests/test_models.py` — model smoke tests
- `tests/test_transforms.py` — transform behaviour
- `tests/test_persistence.py` — save/load round trips
