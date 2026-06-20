"""
CLI entry point for MedAI.
Usage:
  medai train --dataset diabetes_100k
  medai predict --dataset diabetes_100k --features '{"age": 50, "bmi": 28, ...}'
  medai api --port 8000
  medai ui --port 8501
"""

import argparse
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
    import json

    from src.data_loader import DataLoader
    from src.model_trainer import ModelTrainer

    setup_logging(AppConfig.load().raw)

    loader = DataLoader(args.dataset)
    df = loader.load()
    X_train, X_test, y_train, y_test, features = loader.split(df)

    trainer = ModelTrainer()
    trainer.train(X_train, X_test, y_train, y_test, features)

    input_features = json.loads(args.features)
    import pandas as pd

    input_df = pd.DataFrame([input_features])
    result = trainer.predict(input_df)

    label = "POSITIVE" if result["prediction"] == 1 else "NEGATIVE"
    print(f"\nPrediction: {label}")
    print(f"Probability: {result['probability']:.4f}")


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


def main():
    parser = argparse.ArgumentParser(prog="medai", description="MedAI — Enterprise Medical AI")
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train", help="Train models")
    p_train.add_argument(
        "--dataset",
        default="diabetes_100k",
        choices=list(
            __import__("src.data_loader", fromlist=["DataLoader"]).DataLoader.SUPPORTED.keys()
        ),
    )
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="Make prediction")
    p_pred.add_argument("--dataset", default="diabetes_100k")
    p_pred.add_argument("--features", required=True, help='JSON: {"age":50,"bmi":28}')
    p_pred.set_defaults(func=cmd_predict)

    p_api = sub.add_parser("api", help="Start REST API")
    p_api.add_argument("--host", default="0.0.0.0")
    p_api.add_argument("--port", type=int, default=8000)
    p_api.set_defaults(func=cmd_api)

    p_ui = sub.add_parser("ui", help="Start Streamlit dashboard")
    p_ui.add_argument("--port", type=int, default=8501)
    p_ui.set_defaults(func=cmd_ui)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
