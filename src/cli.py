"""
CLI entry point for MedAI v3.
Usage:
  medai train --dataset diabetes_500k
  medai predict --dataset diabetes_500k --features '{"age":50,"bmi":28,...}'
  medai api --port 8000
  medai ui --port 8501
  medai datasets
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AppConfig, setup_logging


def cmd_train(args):
    from src.data_loader import DataLoader
    from src.model_trainer import ModelTrainer

    config = AppConfig.load()
    setup_logging(config.raw)

    loader = DataLoader(args.dataset)
    df = loader.load()
    X_train, X_test, y_train, y_test, features = loader.split(df)

    trainer = ModelTrainer()
    results, models = trainer.train(X_train, X_test, y_train, y_test, features)

    print(f"\n{'=' * 60}")
    print(f"MedAI — Training Results for {args.dataset}")
    print(f"{'=' * 60}")
    print(
        results[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].to_string(
            index=False
        )
    )
    print(f"\nBest: {trainer.best_name} (ROC-AUC: {results.iloc[0]['roc_auc']:.4f})")


def cmd_predict(args):
    import pandas as pd

    from src.data_loader import DataLoader
    from src.model_trainer import ModelTrainer

    setup_logging(AppConfig.load().raw)

    loader = DataLoader(args.dataset)
    df = loader.load()
    X_train, X_test, y_train, y_test, features = loader.split(df)

    trainer = ModelTrainer()
    trainer.train(X_train, X_test, y_train, y_test, features)

    input_features = json.loads(args.features)
    input_df = pd.DataFrame([input_features])
    result = trainer.predict(input_df)

    prob = result["probability"]
    risk = "low" if prob < 0.3 else ("medium" if prob < 0.6 else "high")
    label = "POSITIVE" if result["prediction"] == 1 else "NEGATIVE"
    print(f"\nPrediction: {label}")
    print(f"Probability: {prob:.4f}")
    print(f"Risk level: {risk}")


def cmd_api(args):
    import uvicorn

    setup_logging(AppConfig.load().raw)
    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=False)


def cmd_ui(args):
    import subprocess

    subprocess.run(
        [
            "streamlit",
            "run",
            "app/app.py",
            "--server.port",
            str(args.port),
            "--server.headless",
            "true",
        ]
    )


def cmd_datasets(args):
    from src.data_loader import DataLoader

    print(f"\n{'=' * 60}")
    print("Supported Datasets")
    print(f"{'=' * 60}")
    for name, config in DataLoader.SUPPORTED.items():
        print(f"\n  {name}")
        print(f"    Description: {config.get('description', 'N/A')}")
        print(f"    Target: {config['target']}")
        n_feat = len(config.get("numeric", []) or []) + len(config.get("categorical", []))
        print(f"    Features: {n_feat}")


def main():
    parser = argparse.ArgumentParser(
        prog="medai", description="MedAI v3 — Enterprise Medical AI (500K patients)"
    )
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train", help="Train models on a dataset")
    p_train.add_argument(
        "--dataset",
        default="diabetes_500k",
        choices=list(
            __import__("src.data_loader", fromlist=["DataLoader"]).DataLoader.SUPPORTED.keys()
        ),
    )
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="Make a prediction")
    p_pred.add_argument("--dataset", default="diabetes_500k")
    p_pred.add_argument("--features", required=True, help="JSON dict of features")
    p_pred.set_defaults(func=cmd_predict)

    p_api = sub.add_parser("api", help="Start REST API server")
    p_api.add_argument("--host", default="0.0.0.0")
    p_api.add_argument("--port", type=int, default=8000)
    p_api.set_defaults(func=cmd_api)

    p_ui = sub.add_parser("ui", help="Start Streamlit dashboard")
    p_ui.add_argument("--port", type=int, default=8501)
    p_ui.set_defaults(func=cmd_ui)

    p_ds = sub.add_parser("datasets", help="List supported datasets")
    p_ds.set_defaults(func=cmd_datasets)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
