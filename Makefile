MAIN = 

run:
	python3 $(MAIN) 

install:
	pip install 

debug:
	python3 -m pdb $(NAME)

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

clean:
	rm -rf *__pycache__ */*/__pycache__ __pycache__
	rm -rf .mypy_cache

.PHONY: install run debug lint clean