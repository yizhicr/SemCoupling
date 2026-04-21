import pickle
import json
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from xgboost import XGBClassifier

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.pairwise_features import PairwiseFeatureExtractor


def evaluate_cross_project_prediction(
    source_project: str,
    target_project: str,
    projects_dir: str = "./projects",
    threshold: float = 0.5,
    model_path: str = None,
    scaler_path: str = None
):
    """
    评估跨项目预测的质量
    
    Args:
        source_project: 源项目名称（训练模型的项目）
        target_project: 目标项目名称（被预测的项目）
        projects_dir: 项目根目录
        threshold: 预测概率阈值
        model_path: 自定义模型路径（可选）
        scaler_path: 自定义标准化器路径（可选）
    """
    source_dir = Path(projects_dir) / source_project
    target_dir = Path(projects_dir) / target_project
    
    print("="*80)
    print(f"跨项目预测评估: {source_project} → {target_project}")
    print("="*80)
    
    # ========== 1. 加载模型和标准化器 ==========
    print("\n[1/4] 加载模型和标准化器...")
    
    # 使用自定义路径或默认路径
    if model_path is None:
        model_path = source_dir / "models" / "xgboost.pkl"
    else:
        model_path = Path(model_path)
    
    if scaler_path is None:
        scaler_path = source_dir / "features" / "scaler.pkl"
    else:
        scaler_path = Path(scaler_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f" 模型已加载: {model_path}")
    print(f"标准化器已加载: {scaler_path}")
    
    # ========== 2. 加载目标项目的标签数据 ==========
    print("\n[2/4] 加载目标项目的标签数据...")
    labels_path = target_dir / "labels" / "training_labels_v2.json"
    
    if not labels_path.exists():
        raise FileNotFoundError(f"标签文件不存在: {labels_path}，请先为目标项目生成标签")
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    
    # 兼容两种格式：新格式(test_pairs)和旧格式(test)
    if 'test_pairs' in labels_data:
        test_pairs = labels_data['test_pairs']
    elif 'test' in labels_data:
        test_pairs = labels_data['test']
    else:
        raise ValueError(f"标签文件格式错误，缺少test_pairs或test字段")
    
    print(f"  测试集: {len(test_pairs)} 个函数对")
    
    # 提取真实的标签
    y_true = np.array([pair['label'] for pair in test_pairs])
    print(f"    正样本: {y_true.sum()}, 负样本: {(1 - y_true).sum()}")
    
    # ========== 3. 提取测试集特征并预测 ==========
    print("\n[3/4] 提取测试集特征并进行预测...")
    
    # 初始化特征提取器（使用目标项目的数据）
    feature_extractor = PairwiseFeatureExtractor(
        project_dir=str(target_dir),
        use_codebert=True,
        inference_mode=False  # 需要演化统计特征
    )
    
    # 提取测试集特征
    func1_ids = [pair['func1'] for pair in test_pairs]
    func2_ids = [pair['func2'] for pair in test_pairs]
    
    X_test = []
    for func1_id, func2_id in zip(func1_ids, func2_ids):
        features = feature_extractor._extract_single_pair_features(func1_id, func2_id, {})
        X_test.append(features)
    
    X_test = np.array(X_test)
    print(f"特征提取完成: {X_test.shape}")
    
    # 【方案一】使用目标项目自适配标准化器
    # 从测试集自身拟合标准化器（无监督方式）
    print("\n使用目标项目测试集特征分布计算标准化器...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_test)  # 仅使用测试集特征拟合
    print(f"标准化器已基于目标项目测试集特征拟合完成")
    
    # 标准化特征
    X_test_scaled = scaler.transform(X_test)
    
    # 预测概率
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    print(f"  预测完成")
    print(f"  概率范围: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
    print(f"  平均概率: {y_pred_proba.mean():.4f}")
    
    # ========== 4. 计算评估指标 ==========
    print("\n[4/4] 计算评估指标...")
    
    # AUC-ROC
    try:
        auc_roc = roc_auc_score(y_true, y_pred_proba)
        print(f"\n  AUC-ROC: {auc_roc:.4f}")
    except Exception as e:
        print(f"\n  AUC-ROC 计算失败: {e}")
        auc_roc = None
    
    # AUC-PR
    try:
        auc_pr = average_precision_score(y_true, y_pred_proba)
        print(f"  AUC-PR:  {auc_pr:.4f}")
    except Exception as e:
        print(f"  AUC-PR 计算失败: {e}")
        auc_pr = None
    
    # Precision@K 和 Recall@K
    k_values = [50, 100, 200, 500]
    print(f"\n  Precision@K 和 Recall@K:")
    for k in k_values:
        if k > len(y_pred_proba):
            continue
        
        # 获取Top-K预测的索引
        top_k_indices = np.argsort(y_pred_proba)[-k:]
        y_pred_top_k = y_true[top_k_indices]
        
        precision_at_k = y_pred_top_k.sum() / k
        recall_at_k = y_pred_top_k.sum() / y_true.sum()
        
        print(f"    @{k:4d}: Precision={precision_at_k:.4f}, Recall={recall_at_k:.4f}")
    
    # 混淆矩阵（在给定阈值下）
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    tp = ((y_pred_binary == 1) & (y_true == 1)).sum()
    fp = ((y_pred_binary == 1) & (y_true == 0)).sum()
    tn = ((y_pred_binary == 0) & (y_true == 0)).sum()
    fn = ((y_pred_binary == 0) & (y_true == 1)).sum()
    
    print(f"\n  混淆矩阵 (threshold={threshold}):")
    print(f"    TP={tp}, FP={fp}")
    print(f"    FN={fn}, TN={tn}")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n  分类指标:")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    
    # 泛化能力评估
    print("\n" + "="*80)
    print("泛化能力评估:")
    print("="*80)
    
    if auc_roc is not None:
        if auc_roc > 0.8:
            print(" 优秀")
        elif auc_roc > 0.7:
            print(" 良好")
        elif auc_roc > 0.6:
            print(" 一般")
        else:
            print(" 较差")
    
    return {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="评估跨项目预测的质量")
    parser.add_argument("--source", type=str, required=True, help="源项目名称（训练模型的项目）")
    parser.add_argument("--target", type=str, required=True, help="目标项目名称（被预测的项目）")
    parser.add_argument("--projects-dir", type=str, default="./projects", help="项目根目录")
    parser.add_argument("--threshold", type=float, default=0.5, help="预测概率阈值（默认: 0.5）")
    parser.add_argument("--model-path", type=str, default=None, help="自定义模型路径（可选）")
    parser.add_argument("--scaler-path", type=str, default=None, help="自定义标准化器路径（可选）")
    
    args = parser.parse_args()
    
    results = evaluate_cross_project_prediction(
        source_project=args.source,
        target_project=args.target,
        projects_dir=args.projects_dir,
        threshold=args.threshold,
        model_path=args.model_path,
        scaler_path=args.scaler_path
    )
    
    print("\n 评估完成！")


if __name__ == "__main__":
    main()
