"""Experiment 1: reproducible TF-IDF text classification comparison.

Run: python experiment1.py --data-dir "PATH_TO_data-Project1"
The default data directory is ../data-Project1 relative to this script.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent.parent / 'data-Project1')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.data_dir / 'train_data.csv'
    test_path = args.data_dir / 'test_data_unlabeled.csv'
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    texts = train['text'].fillna('').astype(str)
    labels = train['target']
    test_texts = test['text'].fillna('').astype(str)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    common = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2,
                             max_features=100000, sublinear_tf=True, strip_accents='unicode')
    models = {
        'Linear SVM': (Pipeline([('tfidf', common), ('clf', LinearSVC(random_state=42))]),
                       {'clf__C': [0.1, 1.0, 10.0]}),
        'Logistic Regression': (Pipeline([('tfidf', common), ('clf', LogisticRegression(max_iter=3000, solver='lbfgs', random_state=42))]),
                                {'clf__C': [0.1, 1.0, 10.0]}),
    }
    rows = []
    fitted = {}
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    for name, (pipe, grid) in models.items():
        print(f'Tuning {name} ({args.folds}-fold stratified CV)...', flush=True)
        search = GridSearchCV(pipe, grid, scoring={'accuracy': 'accuracy', 'macro_f1': 'f1_macro'}, refit='accuracy', cv=cv, n_jobs=1,
                              return_train_score=True, verbose=0)
        started = time.time()
        search.fit(texts, labels)
        elapsed = time.time() - started
        fitted[name] = search
        result = pd.DataFrame(search.cv_results_)
        for _, r in result.iterrows():
            rows.append({
                'model': name, 'C': float(r['param_clf__C']),
                'cv_accuracy_mean': float(r['mean_test_accuracy']),
                'cv_accuracy_std': float(r['std_test_accuracy']),
                'cv_train_accuracy_mean': float(r['mean_train_accuracy']),
                'cv_macro_f1_mean': float(r['mean_test_macro_f1']),
                'fit_seconds_grid': round(elapsed, 2),
            })
        print(f'  best={search.best_params_}, accuracy={search.best_score_:.4f}', flush=True)
    metrics = pd.DataFrame(rows).sort_values(['cv_accuracy_mean', 'cv_macro_f1_mean'], ascending=False)
    metrics.to_csv(args.output_dir / 'validation_results.csv', index=False, encoding='utf-8-sig')
    best_row = metrics.iloc[0]
    best_name = str(best_row['model'])
    best_search = fitted[best_name]
    best_model = best_search.best_estimator_
    # GridSearchCV refits on all labeled data; use that selected pipeline for the unlabeled set.
    preds = best_model.predict(test_texts)
    # Match the sample format: one predicted class per line, without a header.
    # The provided template is one row shorter than the unlabeled test set.
    pd.DataFrame(preds).to_csv(args.output_dir / 'prediction.csv', index=False, header=False)
    summary = {
        'training_rows': int(len(train)), 'test_rows': int(len(test)),
        'classes': sorted(pd.Series(labels).unique().tolist()), 'folds': args.folds,
        'best_model': best_name, 'best_params': best_search.best_params_,
        'best_cv_accuracy': float(best_search.best_score_),
        'best_cv_macro_f1': float(best_row['cv_macro_f1_mean']),
        'vectorizer': 'TF-IDF, English stop words, word uni+bi-grams, min_df=2, max_features=100000, sublinear_tf=True',
        'random_state': 42,
    }
    (args.output_dir / 'experiment_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Saved predictions: {args.output_dir / "predictions.csv"}', flush=True)

if __name__ == '__main__':
    main()










