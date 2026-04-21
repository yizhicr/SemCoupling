import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse


def load_project_data(project_name: str, projects_dir: str = "./projects"):
    """
    加载单个项目的训练数据（从.npy文件）
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    project_dir = Path(projects_dir) / project_name
    features_dir = project_dir / "features"
    
    # 检查特征文件是否存在
    required_files = [
        features_dir / "X_train.npy",
        features_dir / "y_train.npy",
        features_dir / "X_val.npy",
        features_dir / "y_val.npy",
        features_dir / "X_test.npy",
        features_dir / "y_test.npy"
    ]
    
    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"特征文件不存在: {file_path}")
    
    # 加载数据
    X_train = np.load(features_dir / "X_train.npy")
    y_train = np.load(features_dir / "y_train.npy")
    X_val = np.load(features_dir / "X_val.npy")
    y_val = np.load(features_dir / "y_val.npy")
    X_test = np.load(features_dir / "X_test.npy")
    y_test = np.load(features_dir / "y_test.npy")
    
    print(f"  {project_name}:")
    print(f"    训练集: {X_train.shape[0]} 样本 (维度: {X_train.shape[1]}), 正样本比例: {y_train.mean():.3f}")
    print(f"    验证集: {X_val.shape[0]} 样本")
    print(f"    测试集: {X_test.shape[0]} 样本")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def joint_training(
    project_names: list,
    projects_dir: str = "./projects",
    output_dir: str = "./model"
):
    """
    多项目联合训练
    
    Args:
        project_names: 项目名称列表
        projects_dir: 项目根目录
        output_dir: 模型输出目录
    """
    print("="*80)
    print("多项目联合训练")
    print("="*80)
    print(f"\n参与项目: {', '.join(project_names)}")
    print(f"项目目录: {projects_dir}")
    print(f"输出目录: {output_dir}\n")
    
    # ========== 1. 加载所有项目的数据 ==========
    print("[1/4] 加载各项目的训练数据...")
    all_X_train = []
    all_y_train = []
    all_X_val = []
    all_y_val = []
    all_X_test = []
    all_y_test = []
    
    for project_name in project_names:
        print(f"\n加载 {project_name} 的数据...")
        X_train, y_train, X_val, y_val, X_test, y_test = load_project_data(
            project_name, projects_dir
        )
        all_X_train.append(X_train)
        all_y_train.append(y_train)
        all_X_val.append(X_val)
        all_y_val.append(y_val)
        all_X_test.append(X_test)
        all_y_test.append(y_test)
    
    # ========== 2. 合并数据 ==========
    print("\n[2/4] 合并所有项目的数据...")
    X_train_all = np.vstack(all_X_train)
    y_train_all = np.hstack(all_y_train)
    
    X_val_all = np.vstack(all_X_val)
    y_val_all = np.hstack(all_y_val)
    
    X_test_all = np.vstack(all_X_test)
    y_test_all = np.hstack(all_y_test)
    
    print(f"  合并后的训练集: {X_train_all.shape[0]} 样本, 维度: {X_train_all.shape[1]}")
    print(f"  合并后的验证集: {X_val_all.shape[0]} 样本")
    print(f"  合并后的测试集: {X_test_all.shape[0]} 样本")
    print(f"  正样本比例: {y_train_all.mean():.3f}")
    
    # ========== 3. 标准化特征 ==========
    print("\n[3/4] 标准化特征...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_all)
    X_val_scaled = scaler.transform(X_val_all)
    X_test_scaled = scaler.transform(X_test_all)
    
    print(f"  ✓ 标准化器已拟合")
    
    # ========== 4. 训练模型 ==========
    print("\n[4/4] 训练XGBoost模型...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='auc',
        use_label_encoder=False,
        n_jobs=-1
    )
    
    model.fit(
        X_train_scaled, y_train_all,
        eval_set=[(X_val_scaled, y_val_all)],
        verbose=True
    )
    
    # 评估模型
    y_val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    val_auc_roc = roc_auc_score(y_val_all, y_val_pred_proba)
    val_auc_pr = average_precision_score(y_val_all, y_val_pred_proba)
    
    test_auc_roc = roc_auc_score(y_test_all, y_test_pred_proba)
    test_auc_pr = average_precision_score(y_test_all, y_test_pred_proba)
    
    print(f"\n{'='*80}")
    print("训练结果汇总")
    print(f"{'='*80}")
    print(f"验证集 AUC-ROC: {val_auc_roc:.4f}")
    print(f"验证集 AUC-PR:  {val_auc_pr:.4f}")
    print(f"测试集 AUC-ROC: {test_auc_roc:.4f}")
    print(f"测试集 AUC-PR:  {test_auc_pr:.4f}")
    
    # 保存模型和标准化器
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = output_path / "xgboost_joint.pkl"
    scaler_path = output_path / "scaler_joint.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\n✓ 模型已保存至: {model_path}")
    print(f"✓ 标准化器已保存至: {scaler_path}")
    
    # 保存项目元信息
    metadata = {
        'project_names': project_names,
        'feature_dim': X_train_all.shape[1],
        'train_samples': X_train_all.shape[0],
        'val_samples': X_val_all.shape[0],
        'test_samples': X_test_all.shape[0],
        'val_auc_roc': float(val_auc_roc),
        'val_auc_pr': float(val_auc_pr),
        'test_auc_roc': float(test_auc_roc),
        'test_auc_pr': float(test_auc_pr)
    }
    
    metadata_path = output_path / "joint_training_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f" 元数据已保存至: {metadata_path}")
    
    return model, scaler


def main():
    parser = argparse.ArgumentParser(description="多项目联合训练")
    parser.add_argument("--projects", type=str, nargs='+', required=True, 
                       help="参与训练的项目名称列表")
    parser.add_argument("--projects-dir", type=str, default="./projects", 
                       help="项目根目录")
    parser.add_argument("--output-dir", type=str, default="./model", 
                       help="模型输出目录")
    
    args = parser.parse_args()
    
    joint_training(
        project_names=args.projects,
        projects_dir=args.projects_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
