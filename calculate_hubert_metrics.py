import pandas as pd
import numpy as np
import os
import argparse
import fnmatch
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr, kendalltau

def calculate_and_print_metrics(df: pd.DataFrame, task_name: str, root_dir: str):
    """
    指定されたDataFrameから評価指標を計算し表示する。
    """
    print(f"\n--- Metrics for task '{task_name}' in directory: {root_dir} ---")

    if task_name not in df.columns or 'predict' not in df.columns:
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
    parser = argparse.ArgumentParser(description="指定されたディレクトリから output.csv ファイルを再帰的に探し、評価指標をまとめて計算します。")
    parser.add_argument('directory', type=str, help="output.csv ファイルを探すルートディレクトリのパス")
    args = parser.parse_args()

    root_dir = args.directory
    if not os.path.isdir(root_dir):
        print(f"エラー: '{root_dir}' は有効なディレクトリではありません。")
        return

    # intelligibility と naturalness のファイルを分けて収集
    tasks = {'intelligibility': [], 'naturalness': []}
    for dirpath, _, filenames in os.walk(root_dir):
        for task_name in tasks.keys():
            if task_name in dirpath:
                for filename in fnmatch.filter(filenames, 'output.csv'):
                    tasks[task_name].append(os.path.join(dirpath, filename))

    if not any(tasks.values()):
        print(f"'{root_dir}' 以下に 'output.csv' ファイルは見つかりませんでした。")
        return

    for task_name, file_list in tasks.items():
        if not file_list:
            continue

        df_list = []
        for file_path in file_list:
            try:
                df_list.append(pd.read_csv(file_path))
            except Exception as e:
                print(f"警告: ファイル '{file_path}' の読み込みに失敗しました: {e}")
        
        if not df_list:
            continue

        combined_df = pd.concat(df_list, ignore_index=True)
        calculate_and_print_metrics(combined_df, task_name, root_dir)

if __name__ == '__main__':
    main()
