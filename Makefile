.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python src/main.py --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json

debug:
	uv run python -m pdb src/main.py $(ARGS)

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache src/__pycache__

lint:
	flake8 src/
	mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 src/
	mypy src/ --strict