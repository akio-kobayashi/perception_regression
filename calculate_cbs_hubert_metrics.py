import pandas as pd
import numpy as np
import os
import argparse
import fnmatch
from sklearn.metrics import mean_absolute_error, accuracy_score
from scipy.stats import spearmanr, kendalltau

def calculate_and_print_metrics(csv_file_path: str):
    """
    指定されたCSVファイルから各タスクの評価指標を計算し表示する。
    """
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"エラー: ファイル '{csv_file_path}' の読み込み中に問題が発生しました: {e}")
        return

    print(f"\n--- Metrics for: {csv_file_path} ---")

    # Ordinal tasks
    ordinal_tasks = {
        'intelligibility': 'predict_int',
        'naturalness': 'predict_nat'
    }
    for task_name, pred_col in ordinal_tasks.items():
        if task_name not in df.columns or pred_col not in df.columns:
            print(f"  タスク '{task_name}': 必要な列が見つかりませんでした。")
            continue

        true_scores = df[task_name].dropna()
        pred_scores = df[pred_col].dropna()

        common_indices = true_scores.index.intersection(pred_scores.index)
        if len(common_indices) < 2:
            print(f"  タスク '{task_name}': 比較可能なデータが2点未満です。")
            continue
        
        true_scores = true_scores.loc[common_indices]
        pred_scores = pred_scores.loc[common_indices]

        try:
            mae = mean_absolute_error(true_scores, pred_scores)
            spearman_rho, _ = spearmanr(true_scores, pred_scores)
            kendall_tau, _ = kendalltau(true_scores, pred_scores)

            print(f"  Task: {task_name}")
            print(f"    MAE: {mae:.4f}")
            print(f"    Spearman's rho: {spearman_rho:.4f}")
            print(f"    Kendall's tau: {kendall_tau:.4f}")
        except Exception as e:
            print(f"  タスク '{task_name}': メトリクス計算中にエラーが発生しました: {e}")

    # Binary classification tasks (CBs)
    for i in range(1, 5):
        true_col = f'cb{i}'
        pred_col = f'predict_cb{i}'
        if true_col not in df.columns or pred_col not in df.columns:
            # This is not an error, the file might just not be from a cbs experiment
            continue

        true_labels = df[true_col].dropna()
        pred_labels = df[pred_col].dropna()

        common_indices = true_labels.index.intersection(pred_labels.index)
        if len(common_indices) == 0:
            continue
            
        true_labels = true_labels.loc[common_indices]
        pred_labels = pred_labels.loc[common_indices]

        try:
            accuracy = accuracy_score(true_labels, pred_labels)
            print(f"  Task: {true_col}")
            print(f"    Accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"  タスク '{true_col}': 精度計算中にエラーが発生しました: {e}")


def main():
    parser = argparse.ArgumentParser(description="指定されたディレクトリから output.csv ファイルを再帰的に探し、評価指標を計算します。")
    parser.add_argument('directory', type=str, help="output.csv ファイルを探すルートディレクトリのパス")
    args = parser.parse_args()

    root_dir = args.directory
    if not os.path.isdir(root_dir):
        print(f"エラー: '{root_dir}' は有効なディレクトリではありません。")
        return

    found_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, 'output.csv'):
            found_files.append(os.path.join(dirpath, filename))

    if not found_files:
        print(f"'{root_dir}' 以下に 'output.csv' ファイルは見つかりませんでした。")
        return

    for file_path in sorted(found_files):
        calculate_and_print_metrics(file_path)

if __name__ == '__main__':
    main()
