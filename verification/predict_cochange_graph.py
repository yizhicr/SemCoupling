import pickle
import json
import sys
import numpy as np
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from itertools import combinations
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.pairwise_features import PairwiseFeatureExtractor  # 复用特征提取器


def build_cochange_prediction_graph(
    project_name: str,
    projects_dir: str = "./projects",
    model_path: str = None,
    threshold: float = 0.7
) -> nx.Graph:
    """
    对新项目生成函数共改预测图
    
    Args:
        project_name: 新项目名称
        projects_dir: 项目根目录
        model_path: 训练好的 XGBoost 模型路径（默认使用 model/xgboost.pkl）
        threshold: 预测概率阈值，高于此值则添加边
    
    Returns:
        共改预测图（NetworkX 无向图，边带 weight 属性）
    """
    project_dir = Path(projects_dir) / project_name
    static_dir = project_dir / "static_analysis"
    
    # ========== 1. 加载静态分析数据 ==========
    print("[1/4] 加载静态分析数据...")
    
    # 检查静态分析文件是否存在
    call_graph_path = static_dir / "call_graph.pkl"
    metadata_path = static_dir / "function_metadata.json"
    
    if not call_graph_path.exists():
        raise FileNotFoundError(f"调用图文件不存在: {call_graph_path}，请先运行静态分析")
    if not metadata_path.exists():
        raise FileNotFoundError(f"函数元数据文件不存在: {metadata_path}，请先运行静态分析")
    
    with open(call_graph_path, 'rb') as f:
        call_graph = pickle.load(f)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        function_metadata = json.load(f)
    
    nodes = list(function_metadata.keys())
    print(f"  函数节点数: {len(nodes)}")
    
    # ========== 2. 初始化特征提取器 ==========
    print("[2/4] 初始化特征提取器...")
    
    # 检查是否存在labels目录，如果不存在则使用inference_mode
    labels_dir = project_dir / "labels"
    inference_mode = not (labels_dir.exists() and (labels_dir / "training_labels_v2.json").exists())
    
    extractor = PairwiseFeatureExtractor(
        project_dir=str(project_dir),
        use_codebert=True,  # 启用CodeBERT以匹配训练时的特征配置
        inference_mode=inference_mode  # 如果没有标签数据，使用推理模式
    )
    
    # ========== 3. 加载模型 ==========
    print("[3/4] 加载模型...")
    
    # 如果未指定模型路径，使用默认的 model/xgboost.pkl
    if model_path is None:
        model_path = Path("model/xgboost.pkl")
    else:
        model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}，请先训练模型或指定正确的模型路径")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"  模型路径: {model_path}")
    
    # ========== 4. 生成所有函数对并准备自适配标准化器 ==========
    print("[4/4] 生成函数对并构建自适配标准化器...")
    pred_graph = nx.Graph()
    pred_graph.add_nodes_from(nodes)
    
    # 为避免组合爆炸，可以限制候选对（例如：只考虑同一文件内的函数对，或调用距离 ≤3 的函数对）
    # 这里演示生成所有同文件函数对 + 调用图上距离 ≤2 的函数对
    candidates = set()
    
    # 从function_metadata构建file_to_funcs映射
    file_to_funcs = {}
    for func_id in nodes:
        # 节点ID格式: {relative_path}::{class_name or ''}::{func_name}
        parts = func_id.split('::')
        if len(parts) >= 1:
            file_path = parts[0]
            if file_path not in file_to_funcs:
                file_to_funcs[file_path] = []
            file_to_funcs[file_path].append(func_id)
    
    # 同文件函数对
    for file_path, funcs in file_to_funcs.items():
        if len(funcs) < 2:
            continue
        for u, v in combinations(funcs, 2):
            candidates.add(tuple(sorted([u, v])))
    
    # 调用图上距离 ≤2 的函数对
    for u in tqdm(nodes, desc="  计算调用图邻居"):
        if u not in call_graph:
            continue
        # 距离1的邻居
        for v in call_graph.neighbors(u):
            candidates.add(tuple(sorted([u, v])))
        # 距离2的邻居
        for v in nx.single_source_shortest_path_length(call_graph, u, cutoff=2):
            if v != u:
                candidates.add(tuple(sorted([u, v])))
    
    print(f"  候选函数对数量: {len(candidates)}")
    
    # 批量提取特征并预测
    batch_size = 500
    candidate_list = list(candidates)
    
    # 【方案一】使用目标项目自适配标准化器
    # 原理：不使用源项目的 scaler，而是用目标项目候选对的特征实时拟合标准化器
    # 优势：避免源项目和目标项目特征分布不匹配的问题
    print("\n 使用目标项目自身特征分布计算标准化器...")
    
    # 采样一部分候选对（如前5000个）计算特征统计量
    sample_size = min(5000, len(candidate_list))
    sample_indices = np.random.choice(len(candidate_list), sample_size, replace=False)
    
    print(f"    从 {len(candidate_list)} 个候选对中采样 {sample_size} 个用于拟合标准化器...")
    
    sample_features = []
    for idx in tqdm(sample_indices, desc="    提取样本特征", leave=False):
        u, v = candidate_list[idx]
        feat = extractor._extract_single_pair_features(u, v, {})
        sample_features.append(feat)
    
    # 拟合标准化器
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(np.array(sample_features))
    print(f"     标准化器已基于目标项目特征拟合完成")
    
    # 用于收集所有预测概率
    all_probs = []
    
    for i in tqdm(range(0, len(candidate_list), batch_size), desc="  预测共改概率"):
        batch = candidate_list[i:i+batch_size]
        features = []
        for u, v in batch:
            feat = extractor._extract_single_pair_features(u, v, {})
            features.append(feat)
        
        X = np.array(features, dtype=np.float32)
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[:, 1]  # 正类概率
        
        # 收集概率用于统计
        all_probs.extend(probs.tolist())
        
        for (u, v), prob in zip(batch, probs):
            if prob >= threshold:
                pred_graph.add_edge(u, v, weight=float(prob))
    
    # 打印概率分布统计
    if all_probs:
        probs_array = np.array(all_probs)
        print(f"\n  [调试] 预测概率统计:")
        print(f"    最小值: {probs_array.min():.6f}")
        print(f"    最大值: {probs_array.max():.6f}")
        print(f"    平均值: {probs_array.mean():.6f}")
        print(f"    中位数: {np.median(probs_array):.6f}")
        print(f"    标准差: {probs_array.std():.6f}")
        print(f"    >0.1的样本数: {(probs_array > 0.1).sum()}")
        print(f"    >0.3的样本数: {(probs_array > 0.3).sum()}")
        print(f"    >0.5的样本数: {(probs_array > 0.5).sum()}")
        print(f"    >0.7的样本数: {(probs_array > 0.7).sum()}")
    
    print(f"  预测图构建完成: {pred_graph.number_of_nodes()} 节点, {pred_graph.number_of_edges()} 边")
    
    return pred_graph


def main():
    import argparse
    parser = argparse.ArgumentParser(description="使用训练好的模型预测新项目的函数共改关系")
    parser.add_argument("--project", type=str, required=True, help="新项目名称")
    parser.add_argument("--projects-dir", type=str, default="./projects", help="项目根目录")
    parser.add_argument("--model-path", type=str, default=None, help="训练好的 XGBoost 模型路径（默认使用 model/xgboost.pkl）")
    parser.add_argument("--threshold", type=float, default=0.7, help="预测概率阈值（默认: 0.7）")
    parser.add_argument("--output", type=str, default=None, help="输出图文件路径 (.pkl)，默认保存到 projects/{project}/cochange_pred_graph.pkl")
    
    args = parser.parse_args()
    
    graph = build_cochange_prediction_graph(
        project_name=args.project,
        projects_dir=args.projects_dir,
        model_path=args.model_path,
        threshold=args.threshold
    )
    
    # 保存图
    output_path = args.output or f"projects/{args.project}/cochange_pred_graph.pkl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(graph, f)
    print(f"\n 共改预测图已保存至: {output_path}")


if __name__ == "__main__":
    main()