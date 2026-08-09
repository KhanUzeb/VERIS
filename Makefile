.PHONY: install install-dev test lint typecheck run dry-run trend diff baseline seed validate usage prune list-cases clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=veris --cov-report=term-missing

lint:
	ruff check veris/ tests/ scripts/

typecheck:
	mypy veris/ --ignore-missing-imports

run:
	@test -n "$(TARGET)" || (echo "set TARGET=module:function" && exit 2)
	python -m veris.cli run --target "$(TARGET)"

dry-run:
	@test -n "$(TARGET)" || (echo "set TARGET=module:function" && exit 2)
	python -m veris.cli run --target "$(TARGET)" --dry-run

list-cases:
	python -m veris.cli list-cases

validate:
	python -m veris.cli validate

trend:
	python -m veris.cli trend

usage:
	python -m veris.cli usage

diff:
	python -m veris.cli diff previous latest

baseline:
	python -m veris.cli baseline

seed:
	@test -n "$(TARGET)" || (echo "set TARGET=module:function" && exit 2)
	python scripts/seed_baseline.py --target "$(TARGET)" --n 3

prune:
	python -m veris.cli prune --days 90

clean:
	python -c "import pathlib, shutil; \
[shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; \
[p.unlink(missing_ok=True) for p in pathlib.Path('.').glob('veris_trends*.db')]; \
[p.unlink(missing_ok=True) for p in pathlib.Path('.').glob('veris_report.*')]; \
[p.unlink(missing_ok=True) for p in pathlib.Path('.').glob('eval_report.*')]"
