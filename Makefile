.PHONY: test smoke ablate-small paper

test:
	pytest -q

smoke:
	python src/dln_core_variable_cycle.py --preset smoke --out outputs/smoke

ablate-small:
	python src/dln_cycle/run_experiments.py --preset ablate-small --out outputs/ablate-small

paper:
	python src/dln_core_variable_cycle.py --preset paper --out outputs/paper
