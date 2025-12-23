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

def cluster_permutation_test(metric_func, y_true, y_pred1, y_pred2, clusters, n_permutations=10000):
    """
    クラスタ単位で置換を行うことで、データ内の相関を考慮した置換検定。
    p値が0になるのを避けるため、ラプラススムージングを適用。
    """
    # nanが含まれている場合、それらを削除して比較
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred1) & ~np.isnan(y_pred2)
    y_true, y_pred1, y_pred2, clusters = y_true[valid_indices], y_pred1[valid_indices], y_pred2[valid_indices], clusters[valid_indices]

    if len(y_true) < 2:
        return np.nan

    metric1 = metric_func(y_true, y_pred1)
    metric2 = metric_func(y_true, y_pred2)
    observed_diff = np.abs(metric1 - metric2)

    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    
    count = 0
    for _ in range(n_permutations):
        # クラスタ単位でスワップするかどうかのマスクを生成
        swap_mask = np.random.rand(n_clusters) < 0.5
        
        perm_pred1 = y_pred1.copy()
        perm_pred2 = y_pred2.copy()
        
        for i, cluster_id in enumerate(unique_clusters):
            if swap_mask[i]:
                cluster_indices = (clusters == cluster_id)
                # このクラスタに属するすべてのサンプルの予測値を入れ替え
                perm_pred1[cluster_indices], perm_pred2[cluster_indices] = perm_pred2[cluster_indices], perm_pred1[cluster_indices]

        perm_metric1 = metric_func(y_true, perm_pred1)
        perm_metric2 = metric_func(y_true, perm_pred2)
        perm_diff = np.abs(perm_metric1 - perm_metric2)

        if perm_diff >= observed_diff:
            count += 1
    
    # p値のゼロ回避 (ラプラススムージング)
    p_value = (count + 1) / (n_permutations + 1)
    return p_value

def holm_bonferroni(p_values):
    """ホルム・ボンフェローニ法によるp値の補正"""
    p_values = np.array(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    
    adjusted_p_values = np.zeros_like(p_values)
    m = len(p_values)
    
    for i, p in enumerate(sorted_p_values):
        adjusted_p = p * (m - i)
        if i > 0:
            adjusted_p = max(adjusted_p, adjusted_p_values[sorted_indices[i-1]])
        adjusted_p_values[sorted_indices[i]] = min(adjusted_p, 1.0)
        
    return adjusted_p_values

def find_csv_files(root_dir, exclude_hearing):
    """ディレクトリから output.csv ファイルを再帰的に検索"""
    file_list = []
    for dirpath, _, filenames in os.walk(root_dir):
        if exclude_hearing and ('BF' in dirpath or 'BM' in dirpath):
            continue
        for filename in fnmatch.filter(filenames, 'output.csv'):
            file_list.append(os.path.join(dirpath, filename))
    return file_list

def load_and_combine_data(file_list):
    """複数のCSVファイルを読み込んで結合"""
    if not file_list: return None
    df_list = [pd.read_csv(f) for f in file_list if os.path.exists(f)]
    return pd.concat(df_list, ignore_index=True) if df_list else None

def main():
    parser = argparse.ArgumentParser(
        description="複数モデルの評価指標について、クラスタ置換検定とホルム法による多重比較補正を用いて有意差を検定します。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--dirs', nargs='+', required=True, help='比較する各モデルの結果が格納されたルートディレクトリのパス')
    parser.add_argument('--task', type=str, required=True, choices=['intelligibility', 'naturalness'], help='評価タスク')
    parser.add_argument('--pred-cols', nargs='+', required=True, help='各ディレクトリに対応する予測値の列名')
    parser.add_argument('--n-permutations', type=int, default=10000, help='置換検定の繰り返し回数')
    parser.add_argument('--exclude-hearing', action='store_true', help="健聴話者('BF'/'BM')を除外する場合に指定")
    
    args = parser.parse_args()

    if len(args.dirs) != len(args.pred_cols):
        raise ValueError("ディレクトリの数と予測列の数は一致させてください。")

    model_dfs = []
    for i, directory in enumerate(args.dirs):
        file_list = find_csv_files(directory, args.exclude_hearing)
        if not file_list:
            print(f"警告: ディレクトリ '{directory}' で 'output.csv' が見つかりませんでした。")
            continue
        
        df = load_and_combine_data(file_list)
        if df is not None:
            df = df.rename(columns={args.pred_cols[i]: f"predict_model_{i}"})
            model_dfs.append(df[['key', 'listener_id', 'speaker', args.task, f"predict_model_{i}"]])

    if len(model_dfs) < 2:
        print("エラー: 比較対象のデータが2モデル分見つかりませんでした。")
        return

    merged_df = model_dfs[0]
    for i in range(1, len(model_dfs)):
        merged_df = pd.merge(merged_df, model_dfs[i], on=['key', 'listener_id', 'speaker'], how='inner')

    y_true = merged_df[args.task].values
    clusters = merged_df['speaker'].values
    
    model_indices = range(len(args.dirs))
    comparisons = list(itertools.combinations(model_indices, 2))
    
    results = []
    for i, j in comparisons:
        model1_name = os.path.basename(args.dirs[i].rstrip('/'))
        model2_name = os.path.basename(args.dirs[j].rstrip('/'))
        y_pred1 = merged_df[f"predict_model_{i}"].values
        y_pred2 = merged_df[f"predict_model_{j}"].values

        p_mae = cluster_permutation_test(calculate_mae, y_true, y_pred1, y_pred2, clusters, args.n_permutations)
        p_rho = cluster_permutation_test(calculate_spearman, y_true, y_pred1, y_pred2, clusters, args.n_permutations)
        p_tau = cluster_permutation_test(calculate_kendall, y_true, y_pred1, y_pred2, clusters, args.n_permutations)
        
        results.append({
            'comparison': f"{model1_name} vs {model2_name}",
            'p_mae': p_mae, 'p_rho': p_rho, 'p_tau': p_tau
        })

    # 多重比較補正
    p_values_mae = [r['p_mae'] for r in results]
    p_values_rho = [r['p_rho'] for r in results]
    p_values_tau = [r['p_tau'] for r in results]

    adj_p_mae = holm_bonferroni(p_values_mae)
    adj_p_rho = holm_bonferroni(p_values_rho)
    adj_p_tau = holm_bonferroni(p_values_tau)

    for i, r in enumerate(results):
        r['adj_p_mae'] = adj_p_mae[i]
        r['adj_p_rho'] = adj_p_rho[i]
        r['adj_p_tau'] = adj_p_tau[i]

    # 結果の表示
    print("\n" + "="*70)
    print(f"タスク '{args.task}' の有意差検定結果 (Holm補正済みp値)")
    print("="*70)
    print(f"{'Comparison':<40} {'MAE p-value':<15} {'Spearman p-value':<20} {'Kendall p-value':<15}")
    print("-"*70)
    for r in results:
        print(f"{r['comparison']:<40} {r['adj_p_mae']:<15.4f} {r['adj_p_rho']:<20.4f} {r['adj_p_tau']:<15.4f}")
    print("-"*70)

if __name__ == '__main__':
    main()