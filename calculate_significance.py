import pandas as pd
import numpy as np
import os
import argparse
import fnmatch
from scipy import stats
import itertools

def calculate_mae(y_true, y_pred):
    """平均絶対誤差（MAE）を計算"""
    return np.mean(np.abs(y_true - y_pred))

def calculate_spearman(y_true, y_pred):
    """スピアマンの順位相関係数を計算"""
    return stats.spearmanr(y_true, y_pred)[0]

def calculate_kendall(y_true, y_pred):
    """ケンドールの順位相関係数を計算"""
    return stats.kendalltau(y_true, y_pred)[0]

def permutation_test(metric_func, y_true, y_pred1, y_pred2, n_permutations=10000):
    """
    2つのモデルから得られた評価指標の差の有意性を評価するための置換検定
    """
    # nanが含まれている場合、それらを削除して比較
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred1) & ~np.isnan(y_pred2)
    y_true = y_true[valid_indices]
    y_pred1 = y_pred1[valid_indices]
    y_pred2 = y_pred2[valid_indices]

    if len(y_true) < 2:
        return np.nan

    metric1 = metric_func(y_true, y_pred1)
    metric2 = metric_func(y_true, y_pred2)
    observed_diff = np.abs(metric1 - metric2)

    count = 0
    for _ in range(n_permutations):
        mask = np.random.rand(len(y_pred1)) < 0.5
        perm_pred1 = np.where(mask, y_pred1, y_pred2)
        perm_pred2 = np.where(mask, y_pred2, y_pred1)

        perm_metric1 = metric_func(y_true, perm_pred1)
        perm_metric2 = metric_func(y_true, perm_pred2)
        perm_diff = np.abs(perm_metric1 - perm_metric2)

        if perm_diff >= observed_diff:
            count += 1

    p_value = count / n_permutations
    return p_value

def find_csv_files(root_dir, exclude_hearing):
    """指定されたタスクのoutput.csvファイルをディレクトリから検索"""
    file_list = []
    for dirpath, _, filenames in os.walk(root_dir):
        # exclude_hearingがTrueの場合、健聴話者を除外
        if exclude_hearing and ('BF' in dirpath or 'BM' in dirpath):
            continue
        for filename in fnmatch.filter(filenames, 'output.csv'):
            file_list.append(os.path.join(dirpath, filename))
    return file_list

def load_and_combine_data(file_list):
    """複数のCSVファイルを読み込んで結合"""
    if not file_list:
        return None
    
    df_list = []
    for f in file_list:
        try:
            df_list.append(pd.read_csv(f))
        except Exception as e:
            print(f"警告: ファイル '{f}' の読み込みに失敗しました: {e}")
            
    return pd.concat(df_list, ignore_index=True) if df_list else None

def main():
    parser = argparse.ArgumentParser(
        description="複数のモデルの予測結果について、評価指標の有意差検定を行います。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--dirs', nargs='+', required=True, help='比較する各モデルの結果が格納されたルートディレクトリのパス')
    parser.add_argument('--task', type=str, required=True, choices=['intelligibility', 'naturalness'], help='評価タスク（"intelligibility" または "naturalness"）')
    parser.add_argument('--pred-cols', nargs='+', required=True, help='各ディレクトリに対応する予測値の列名')
    parser.add_argument('--n-permutations', type=int, default=10000, help='置換検定の繰り返し回数')
    parser.add_argument('--exclude-hearing', action='store_true', help="パスに 'BF' または 'BM' を含む話者（健聴話者）を除外する場合に指定")
    
    args = parser.parse_args()

    if len(args.dirs) != len(args.pred_cols):
        raise ValueError("ディレクトリの数と予測列の数は一致させてください。")

    # 各モデルのデータをロード
    model_dfs = []
    for i, directory in enumerate(args.dirs):
        file_list = find_csv_files(directory, args.exclude_hearing)
        if not file_list:
            print(f"警告: ディレクトリ '{directory}' 内で 'output.csv' が見つかりませんでした。")
            continue
        
        df = load_and_combine_data(file_list)
        if df is not None:
            # 予測列の名前をモデルごとに一意に変更
            df = df.rename(columns={args.pred_cols[i]: f"predict_model_{i}"})
            model_dfs.append(df[['key', 'listener_id', args.task, f"predict_model_{i}"]])

    if len(model_dfs) < 2:
        print("エラー: 比較対象のデータが2モデル分見つかりませんでした。")
        return

    # 'key'と'listener_id'を基準に全モデルのデータをマージ
    merged_df = model_dfs[0]
    for i in range(1, len(model_dfs)):
        merged_df = pd.merge(merged_df, model_dfs[i], on=['key', 'listener_id', args.task], how='inner')

    y_true = merged_df[args.task]

    # 全てのモデルペアで比較検定
    model_indices = range(len(args.dirs))
    for i, j in itertools.combinations(model_indices, 2):
        model1_name = os.path.basename(args.dirs[i].rstrip('/'))
        model2_name = os.path.basename(args.dirs[j].rstrip('/'))
        
        y_pred1 = merged_df[f"predict_model_{i}"]
        y_pred2 = merged_df[f"predict_model_{j}"]

        print("\n" + "="*60)
        print(f"比較: {model1_name} (Model {i+1}) vs {model2_name} (Model {j+1})")
        print("="*60)

        # MAE
        mae1 = calculate_mae(y_true, y_pred1)
        mae2 = calculate_mae(y_true, y_pred2)
        errors1 = np.abs(y_true - y_pred1)
        errors2 = np.abs(y_true - y_pred2)
        _, p_mae = stats.wilcoxon(errors1, errors2, alternative='two-sided')
        print(f"MAE: Model {i+1}={mae1:.4f}, Model {j+1}={mae2:.4f}")
        print(f"  -> p-value (Wilcoxon): {p_mae:.4f}")

        # Spearman
        rho1 = calculate_spearman(y_true, y_pred1)
        rho2 = calculate_spearman(y_true, y_pred2)
        p_rho = permutation_test(calculate_spearman, y_true.values, y_pred1.values, y_pred2.values, args.n_permutations)
        print(f"Spearman rho: Model {i+1}={rho1:.4f}, Model {j+1}={rho2:.4f}")
        print(f"  -> p-value (Permutation): {p_rho:.4f}")

        # Kendall
        tau1 = calculate_kendall(y_true, y_pred1)
        tau2 = calculate_kendall(y_true, y_pred2)
        p_tau = permutation_test(calculate_kendall, y_true.values, y_pred1.values, y_pred2.values, args.n_permutations)
        print(f"Kendall tau: Model {i+1}={tau1:.4f}, Model {j+1}={tau2:.4f}")
        print(f"  -> p-value (Permutation): {p_tau:.4f}")

if __name__ == '__main__':
    main()
