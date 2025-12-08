import pandas as pd
import numpy as np
import os
import argparse
import fnmatch
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr, kendalltau

def calculate_and_print_metrics(csv_file_path: str):
    """
    指定されたCSVファイルから評価指標を計算し表示する。
    ディレクトリ名からタスクを推測する。
    """
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"エラー: ファイル '{csv_file_path}' の読み込み中に問題が発生しました: {e}")
        return

    print(f"\n--- Metrics for: {csv_file_path} ---")

    # ディレクトリパスからタスク名 (intelligibility or naturalness) を推測
    task_name = None
    if 'intelligibility' in csv_file_path:
        task_name = 'intelligibility'
    elif 'naturalness' in csv_file_path:
        task_name = 'naturalness'

    if not task_name or task_name not in df.columns or 'predict' not in df.columns:
        print(f"  評価可能なタスクが見つからないか、必要な列 ('{task_name}', 'predict') がありません。")
        return

    true_scores = df[task_name].dropna()
    pred_scores = df['predict'].dropna()

    common_indices = true_scores.index.intersection(pred_scores.index)
    if len(common_indices) == 0:
        print(f"  タスク '{task_name}': 比較可能なデータがありません。")
        return
    
    true_scores = true_scores.loc[common_indices]
    pred_scores = pred_scores.loc[common_indices]

    if len(true_scores) < 2:
        print(f"  タスク '{task_name}': データ点が少なすぎるため、メトリクスを計算できません。")
        return

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
