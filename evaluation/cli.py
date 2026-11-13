"""Usage: python -m evaluation.cli --config configs/baseline.yaml"""
from __future__ import annotations
import argparse
from .config import load_config
from .runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description='NEP evaluation')
    parser.add_argument('--config', required=True, help='YAML/JSON experiment config')
    args = parser.parse_args()
    _, detailed, summary = run_experiment(load_config(args.config))
    print(f'Scored {len(detailed)} model-task pairs')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
