import argparse
import numpy as np
import pickle
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)


def train_xgboost(project: str = "MaaAssistantArknights", projects_dir: str = "projects"):
    """
    训练 XGBoost 模型用于函数共改预测
    
    Args:
        project: 项目名称
        projects_dir: 项目目录路径
    """
    # 路径配置
    features_dir = Path(projects_dir) / project / "features"
    model_dir = Path(projects_dir) / project / "models"
    
    # 检查特征文件是否存在
    if not features_dir.exists():
        raise FileNotFoundError(f"特征目录不存在: {features_dir}")
    
    # 加载数据
    print("=" * 60)
    print(f"加载 {project} 项目的特征数据")
    print("=" * 60)
    
    X_train = np.load(features_dir / "X_train.npy")
    y_train = np.load(features_dir / "y_train.npy")
    X_val   = np.load(features_dir / "X_val.npy")
    y_val   = np.load(features_dir / "y_val.npy")
    X_test  = np.load(features_dir / "X_test.npy")
    y_test  = np.load(features_dir / "y_test.npy")
    
    print(f"\n数据集统计:")
    print(f"  训练集: {X_train.shape[0]} 样本 (正样本: {y_train.sum()}, 负样本: {(y_train==0).sum()})")
    print(f"  验证集: {X_val.shape[0]} 样本 (正样本: {y_val.sum()}, 负样本: {(y_val==0).sum()})")
    print(f"  测试集: {X_test.shape[0]} 样本 (正样本: {y_test.sum()}, 负样本: {(y_test==0).sum()})")
    
    # 计算类别权重（处理不平衡问题）
    n_pos = y_train.sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"\n类别权重: scale_pos_weight = {scale_pos_weight:.2f}")
    
    # 训练模型
    print("\n" + "=" * 60)
    print("开始训练 XGBoost 模型")
    print("=" * 60)
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # 类别权重，处理不平衡
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1  # 使用所有CPU核心
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    # 评估测试集
    print("\n" + "=" * 60)
    print("测试集评估结果")
    print("=" * 60)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 详细分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['负样本', '正样本']))
    
    # AUC-ROC
    auc_score = roc_auc_score(y_test, y_proba)
    print(f"AUC-ROC: {auc_score:.4f}")
    
    # AUC-PR (更适合不平衡数据)
    ap_score = average_precision_score(y_test, y_proba)
    print(f"AUC-PR:  {ap_score:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n混淆矩阵:")
    print(cm)
    print(f"  TN={cm[0][0]}, FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}, TP={cm[1][1]}")
    
    # Precision-Recall 曲线关键点
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    # 避免除以零错误
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1])
        best_idx = np.argmax(f1_scores)
        if best_idx < len(thresholds):
             print(f"\nPrecision-Recall 关键点:")
             print(f"  最佳 F1 阈值: {thresholds[best_idx]:.4f}")
    
    # 保存模型
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "xgboost.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"\n✓ 模型已保存至: {model_path}")
    
    return model, auc_score, ap_score


def main():
    parser = argparse.ArgumentParser(
        description="训练 XGBoost 模型用于函数共改预测"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="MaaAssistantArknights",
        help="项目名称（默认: MaaAssistantArknights）"
    )
    parser.add_argument(
        "--projects-dir",
        type=str,
        default="projects",
        help="项目目录路径（默认: projects）"
    )
    args = parser.parse_args()
    
    train_xgboost(
        project=args.project,
        projects_dir=args.projects_dir
    )


if __name__ == "__main__":
    main()