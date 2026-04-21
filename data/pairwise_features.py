import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 全局常量
CODEBERT_DIM = 768

FEATURE_NAMES = [
    # 语义特征
    # 注意：实际特征向量由 static_feats (拼接+差值+点积) + structural_feats + evolve_feats 组成
    
    # 粗粒度结构特征 (3个)
    "is_same_file",
    "is_same_directory",
    "directory_depth_diff",
    
    # 演化特征 (2个)
    "func1_modify_count",
    "func2_modify_count",
]


class PairwiseFeatureExtractor:
    """
    成对特征提取器
    
    为每个函数对计算多维度特征向量，包括：
    - 语义特征：CodeBERT嵌入相似度、TF-IDF文本相似度
    - 结构特征：调用图拓扑特征
    - 文件耦合特征：文件位置和历史共改
    - 演化特征：修改历史和作者信息
    - 代码度量特征：函数大小、参数等
    """
    
    def __init__(self, project_dir: str, use_codebert: bool = True, inference_mode: bool = False):
        """
        初始化特征提取器
        
        Args:
            project_dir: 项目目录路径（如 projects/Python）
            use_codebert: 是否使用 CodeBERT 嵌入
            inference_mode: 推理模式，跳过加载标签数据和演化统计
        """
        self.project_dir = Path(project_dir)
        self.use_codebert = use_codebert
        self.inference_mode = inference_mode
        
        print("="*80)
        print(f"初始化特征提取器 - 项目: {self.project_dir.name}")
        if self.inference_mode:
            print("模式: 推理模式 (inference_mode=True)")
        print("="*80)
        
        # ==================== 加载第一阶段静态特征 ====================
        print("\n[1/5] 加载静态特征矩阵...")
        
        static_features_path = self.project_dir / "features" / "function_features.npy"
        static_ids_path = self.project_dir / "features" / "function_ids.json"
        
        if static_features_path.exists() and static_ids_path.exists():
            self.static_features = np.load(static_features_path)
            with open(static_ids_path, 'r', encoding='utf-8') as f:
                self.static_ids = json.load(f)
            self.id_to_index = {nid: i for i, nid in enumerate(self.static_ids)}
            print(f"  [OK] 静态特征矩阵: {self.static_features.shape}")
        else:
            raise FileNotFoundError(
                f"静态特征文件不存在，请先运行第一阶段特征提取。\n"
                f"  期望路径: {static_features_path}"
            )
        
        # ==================== 加载第二阶段演化统计 ====================
        print("\n[2/5] 加载演化统计...")
        
        if not self.inference_mode:
            evolution_stats_path = self.project_dir / "labels" / "function_evolution_stats.json"
            if evolution_stats_path.exists():
                with open(evolution_stats_path, 'r', encoding='utf-8') as f:
                    self.evolution_stats = json.load(f)
                print(f"   演化统计: {len(self.evolution_stats)} 个函数")
            else:
                self.evolution_stats = {}
                print("   function_evolution_stats.json 不存在，演化特征将受限")
            
            # ==================== 加载第三阶段标签数据 ====================
            print("\n[3/5] 加载标签数据...")
            
            labels_path = self.project_dir / "labels" / "training_labels_v2.json"
            if not labels_path.exists():
                raise FileNotFoundError(f"训练标签不存在: {labels_path}")
            
            with open(labels_path, 'r', encoding='utf-8') as f:
                self.labels_data = json.load(f)
            
            self.train_pairs = self.labels_data['train']
            self.val_pairs = self.labels_data['val']
            self.test_pairs = self.labels_data['test']
            
            print(f"   训练集: {len(self.train_pairs)} 样本")
            print(f"   验证集: {len(self.val_pairs)} 样本")
            print(f"   测试集: {len(self.test_pairs)} 样本")
            
            # 加载 commit_to_functions
            commit_to_funcs_path = self.project_dir / "labels" / "commit_to_functions.json"
            if commit_to_funcs_path.exists():
                with open(commit_to_funcs_path, 'r', encoding='utf-8') as f:
                    self.commit_to_functions = json.load(f)
            else:
                self.commit_to_functions = {}
        else:
            # 推理模式下，演化统计和标签数据都初始化为空
            print("  ℹ 推理模式：跳过演化统计加载")
            self.evolution_stats = {}
            self.labels_data = None
            self.train_pairs = []
            self.val_pairs = []
            self.test_pairs = []
            self.commit_to_functions = {}
        
        # ==================== 预计算辅助数据 ====================
        print("\n[4/5] 预计算辅助数据...")
        
        # 加载调用图（用于结构特征）
        call_graph_path = self.project_dir / "static_analysis" / "call_graph.pkl"
        if call_graph_path.exists():
            with open(call_graph_path, 'rb') as f:
                self.call_graph = pickle.load(f)
            print(f"   调用图: {len(self.call_graph.nodes())} 节点, {len(self.call_graph.edges())} 边")
            
            # 计算 PageRank
            self.pagerank_dict = nx.pagerank(self.call_graph)
            print(f"   PageRank 计算完成")
        else:
            self.call_graph = nx.DiGraph()
            self.pagerank_dict = {}
            print("   调用图不存在，结构特征将受限")
        
        # 加载函数元数据（用于代码度量特征）
        metadata_path = self.project_dir / "static_analysis" / "function_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.function_metadata = json.load(f)
            print(f"   函数元数据: {len(self.function_metadata)} 个函数")
        else:
            self.function_metadata = {}
            print("   函数元数据不存在，代码度量特征将受限")
        
        # 初始化文件共改矩阵为空
        # 在推理模式下，file_cochange_freq 将被置零
        self.file_cochange_matrix = {}
        
        # 在推理模式下，func_modify_stats 也初始化为空
        if self.inference_mode:
            self.func_modify_stats = {}
        else:
            # 从演化统计中提取函数修改统计
            self.func_modify_stats = {}
            if self.evolution_stats:
                for func_id, stats in self.evolution_stats.items():
                    # 处理两种可能的数据结构
                    if isinstance(stats, dict):
                        # 新格式: {modify_count, authors, commit_times}
                        self.func_modify_stats[func_id] = {
                            'modify_count': stats.get('modify_count', 0),
                            'authors': set(stats.get('authors', [])),
                            'commit_times': stats.get('commit_times', [])
                        }
                    elif isinstance(stats, (int, float)):
                        # 旧格式: 只是一个计数
                        self.func_modify_stats[func_id] = {
                            'modify_count': int(stats),
                            'authors': set(),
                            'commit_times': []
                        }
                    else:
                        # 未知格式，使用默认值
                        self.func_modify_stats[func_id] = {
                            'modify_count': 0,
                            'authors': set(),
                            'commit_times': []
                        }
        
        # ==================== 初始化嵌入器 ====================
        print("\n[5/5] 初始化嵌入器...")
        
        # TF-IDF 向量化器（用于文本相似度，如果静态特征中未包含）
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        
        # CodeBERT 嵌入器（可选，如果静态特征中未包含）
        self.codebert_cache = {}
        if self.use_codebert:
            try:
                from data.code_embedder import CodeBERTEmbedder
                self.codebert_embedder = CodeBERTEmbedder()
                print(f"   CodeBERT 嵌入器已加载 (维度={CODEBERT_DIM})")
                
                # 尝试加载缓存
                cache_path = self.project_dir / "labels" / "codebert_cache.pkl"
                if cache_path.exists():
                    with open(cache_path, 'rb') as f:
                        self.codebert_cache = pickle.load(f)
                    print(f"   加载 CodeBERT 缓存: {len(self.codebert_cache)} 个函数")
            except Exception as e:
                print(f"   CodeBERT 加载失败: {e}，将使用零向量代替")
                self.use_codebert = False
                self.codebert_embedder = None
        else:
            print("  ℹ CodeBERT 已禁用")
            self.codebert_embedder = None
        
        print("\n" + "="*80)
        print("特征提取器初始化完成！")
        print("="*80)
    
    def _build_file_cochange_matrix(self) -> Dict[Tuple[str, str], int]:
        """
        构建文件级共改矩阵
        
        Returns:
            {(file1, file2): cochange_count}
        """
        file_cochange = defaultdict(int)
        
        for commit_hash, func_ids in self.commit_to_functions.items():
            # 获取这些函数所属的文件
            files_in_commit = set()
            for func_id in func_ids:
                if func_id in self.function_metadata:
                    file_path = self.function_metadata[func_id].get('file_path', '')
                    if file_path:
                        files_in_commit.add(file_path)
            
            # 生成文件对
            files_list = list(files_in_commit)
            for i in range(len(files_list)):
                for j in range(i + 1, len(files_list)):
                    pair = tuple(sorted([files_list[i], files_list[j]]))
                    file_cochange[pair] += 1
        
        return dict(file_cochange)
    
    def _compute_function_modify_stats(self) -> Dict[str, Dict]:
        """
        计算每个函数的修改统计信息
        
        Returns:
            {
                node_id: {
                    'modify_count': int,
                    'authors': set,
                    'timestamps': [str]
                }
            }
        """
        stats = defaultdict(lambda: {
            'modify_count': 0,
            'authors': set(),
            'timestamps': []
        })
        
        # 这里需要从 commits_details 中提取，但为了简化，我们使用 commit_to_functions
        # 如果需要更详细的作者信息，需要在第三阶段保存 commits_details
        
        for commit_hash, func_ids in self.commit_to_functions.items():
            for func_id in func_ids:
                stats[func_id]['modify_count'] += 1
        
        return dict(stats)
    
    def extract_all_features(self, output_dir: str) -> None:
        """
        提取所有样本的特征并保存
        
        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("开始特征提取")
        print("="*80)
        
        # 提取各集合特征
        print("\n[1/3] 提取训练集特征...")
        X_train, y_train, train_ids = self._extract_dataset_features(self.train_pairs, "训练集")
        
        print("\n[2/3] 提取验证集特征...")
        X_val, y_val, val_ids = self._extract_dataset_features(self.val_pairs, "验证集")
        
        print("\n[3/3] 提取测试集特征...")
        X_test, y_test, test_ids = self._extract_dataset_features(self.test_pairs, "测试集")
        
        # ==================== 特征标准化 ====================
        print("\n标准化特征...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)  # 在训练集上拟合
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # ==================== 保存结果 ====================
        print("\n保存特征数据...")
        
        np.save(output_path / "X_train.npy", X_train_scaled)
        np.save(output_path / "y_train.npy", y_train)
        np.save(output_path / "X_val.npy", X_val_scaled)
        np.save(output_path / "y_val.npy", y_val)
        np.save(output_path / "X_test.npy", X_test_scaled)
        np.save(output_path / "y_test.npy", y_test)
        
        with open(output_path / "scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        
        sample_ids = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }
        with open(output_path / "sample_ids.json", 'w', encoding='utf-8') as f:
            json.dump(sample_ids, f, ensure_ascii=False, indent=2)
        
        # 保存 CodeBERT 缓存
        if self.use_codebert and self.codebert_cache:
            cache_path = output_path / "codebert_cache.pkl"
            with open(cache_path, 'wb') as f:
                pickle.dump(self.codebert_cache, f)
            print(f"   CodeBERT 缓存已保存: {len(self.codebert_cache)} 个函数")
        
        # ==================== 打印统计信息 ====================
        print("\n" + "="*80)
        print("特征提取完成！")
        print("="*80)
        print(f"\n特征维度: {X_train.shape[1]}")
        print(f"训练集: {X_train.shape[0]} 样本, 正样本={y_train.sum()}, 负样本={len(y_train)-y_train.sum()}")
        print(f"验证集: {X_val.shape[0]} 样本, 正样本={y_val.sum()}, 负样本={len(y_val)-y_val.sum()}")
        print(f"测试集: {X_test.shape[0]} 样本, 正样本={y_test.sum()}, 负样本={len(y_test)-y_test.sum()}")
        print(f"\n输出目录: {output_path}")
        print(f"  - X_train.npy, y_train.npy")
        print(f"  - X_val.npy, y_val.npy")
        print(f"  - X_test.npy, y_test.npy")
        print(f"  - scaler.pkl")
        print(f"  - sample_ids.json")
    
    def _extract_dataset_features(
        self, 
        pairs: List[Dict], 
        dataset_name: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        提取数据集的特征
        
        Args:
            pairs: 样本对列表
            dataset_name: 数据集名称（用于进度条）
            
        Returns:
            (X, y, sample_ids)
        """
        features_list = []
        labels_list = []
        sample_ids = []
        
        for pair in tqdm(pairs, desc=f"提取{dataset_name}特征"):
            func1_id = pair['func1']
            func2_id = pair['func2']
            label = pair['label']
            metadata = pair.get('metadata', {})
            
            # 提取特征向量
            feature_vec = self._extract_single_pair_features(func1_id, func2_id, metadata)
            
            features_list.append(feature_vec)
            labels_list.append(label)
            sample_ids.append(f"{func1_id}|||{func2_id}")
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        return X, y, sample_ids
    
    def _extract_single_pair_features(
        self, 
        func1_id: str, 
        func2_id: str, 
        pair_metadata: Dict
    ) -> np.ndarray:
        """
        提取单个函数对的完整特征向量
        
        Args:
            func1_id: 函数1的节点ID
            func2_id: 函数2的节点ID
            pair_metadata: 第三阶段提供的元数据（正样本包含 cochange_count 等）
            
        Returns:
            一维特征向量（仅基于纯语义和代码度量特征，移除所有结构相关特征）
        """
        # 获取静态特征向量
        idx1 = self.id_to_index.get(func1_id)
        idx2 = self.id_to_index.get(func2_id)
        
        if idx1 is None or idx2 is None:
            # 如果任一函数不在静态特征中，返回零向量
            static_dim = self.static_features.shape[1]
            static_feats = np.zeros(static_dim * 4)  # 拼接+差值+点积需要4倍维度
        else:
            v1 = self.static_features[idx1]
            v2 = self.static_features[idx2]
            # 特征组合方式：拼接 + 差值 + 点积
            diff = np.abs(v1 - v2)
            product = v1 * v2
            static_feats = np.concatenate([v1, v2, diff, product])
        
        # 【彻底修复】移除所有结构相关特征（包括粗粒度文件特征）
        # 原因：同文件/同目录与调用距离高度相关，会导致数据泄露
        # structural_feats = self._extract_structural_features(func1_id, func2_id)
        structural_feats = [0.0, 0.0, 0.0]  # 强制置零
        
        # 演化特征（仅从 evolution_stats 获取，避免使用 pair_metadata 导致数据泄露）
        evolve_feats = self._extract_evolution_features(func1_id, func2_id)
        
        # 最终特征向量
        return np.concatenate([static_feats, structural_feats, evolve_feats])
    
    def _extract_semantic_features(self, func1_id: str, func2_id: str) -> List[float]:
        """
        提取语义特征（已废弃，静态特征中已包含）
        
        Returns:
            [codebert_cosine_sim, tfidf_cosine_sim]
        """
        # 此方法已废弃，静态特征矩阵中已包含语义信息
        return [0.0, 0.0]
    
    def _get_codebert_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """
        获取或计算 CodeBERT 嵌入（带缓存）
        
        Returns:
            768维嵌入向量，失败返回 None
        """
        if node_id in self.codebert_cache:
            return self.codebert_cache[node_id]
        
        if node_id not in self.function_metadata:
            return None
        
        body_code = self.function_metadata[node_id].get('body_code', '')
        if not body_code or not body_code.strip():
            return None
        
        try:
            embedding = self.codebert_embedder.embed_function(body_code)
            emb_np = embedding.numpy()
            self.codebert_cache[node_id] = emb_np
            return emb_np
        except Exception as e:
            print(f"警告: CodeBERT 嵌入失败 {node_id}: {e}")
            return None
    
    def _extract_structural_features(self, func1_id: str, func2_id: str) -> List[float]:
        """
        提取粗粒度结构特征（不会泄露调用距离信息）
        
        Returns:
            [is_same_file, is_same_directory, directory_depth_diff]
        """
        # 从函数元数据中提取文件路径信息
        meta1 = self.function_metadata.get(func1_id, {})
        meta2 = self.function_metadata.get(func2_id, {})
        
        file_path1 = meta1.get('file_path', '')
        file_path2 = meta2.get('file_path', '')
        
        # 特征1: 是否同文件 (1维)
        is_same_file = 1.0 if file_path1 == file_path2 and file_path1 != '' else 0.0
        
        # 特征2: 是否同目录 (1维)
        dir1 = os.path.dirname(file_path1) if file_path1 else ''
        dir2 = os.path.dirname(file_path2) if file_path2 else ''
        is_same_dir = 1.0 if dir1 == dir2 and dir1 != '' else 0.0
        
        # 特征3: 目录深度差异 (1维)
        depth1 = len(Path(file_path1).parts) if file_path1 else 0
        depth2 = len(Path(file_path2).parts) if file_path2 else 0
        depth_diff = abs(depth1 - depth2)
        
        return [is_same_file, is_same_dir, float(depth_diff)]
    
    def _extract_file_coupling_features(self, func1_id: str, func2_id: str) -> List[float]:
        """
        提取文件耦合特征（已废弃，静态特征中已包含）
        
        Returns:
            [same_file, same_directory, path_prefix_similarity, file_cochange_freq]
        """
        # 此方法已废弃，静态特征矩阵中已包含文件耦合信息
        return [0.0, 0.0, 0.0, 0.0]
    
    def _extract_evolution_features(
        self, 
        func1_id: str, 
        func2_id: str
    ) -> List[float]:
        """
        提取演化特征
        
        Returns:
            [func1_modify_count, func2_modify_count]
            注意：移除了所有来自 pair_metadata 的特征以避免数据泄露
            - cochange_count: 直接来自标签生成，正样本>=阈值，负样本=0
            - avg_time_gap: 仅正样本有此字段，负样本不存在
            - author_jaccard_overlap: 仅正样本有作者信息
        """
        # 推理模式下，所有演化特征置零
        if self.inference_mode:
            return [0.0] * 2
        
        # 特征1&2: 函数修改次数（从 evolution_stats 获取）
        # 这些特征是安全的，因为它们是函数的独立属性，不依赖于函数对
        func1_modify_count = self.evolution_stats.get(func1_id, 0)
        func2_modify_count = self.evolution_stats.get(func2_id, 0)
        
        # 注意：移除了以下会泄露标签的特征：
        # - cochange_count: 直接决定标签
        # - avg_time_gap: 仅正样本有值
        # - author_jaccard_overlap: 仅正样本有作者信息
        
        return [
            float(func1_modify_count),
            float(func2_modify_count)
        ]
    
    def _extract_code_metric_features(self, func1_id: str, func2_id: str) -> List[float]:
        """
        提取代码度量特征（已废弃，静态特征中已包含）
        
        Returns:
            [size_ratio, param_count_ratio, complexity_ratio, loc_diff_normalized]
        """
        # 此方法已废弃，静态特征矩阵中已包含代码度量信息
        return [0.0, 0.0, 0.0, 0.0]


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="第四阶段：多模态特征工程")
    parser.add_argument("--project", type=str, required=True, help="项目名称")
    parser.add_argument("--projects-dir", type=str, default="./projects", help="项目根目录")
    parser.add_argument("--output-dir", type=str, default=None, help="特征输出目录")
    parser.add_argument("--no-codebert", action="store_true", help="禁用 CodeBERT")
    
    args = parser.parse_args()
    
    # 构造路径
    project_dir = Path(args.projects_dir) / args.project
    output_dir = args.output_dir or str(project_dir / "features")
    
    # 初始化特征提取器
    extractor = PairwiseFeatureExtractor(
        project_dir=str(project_dir),
        use_codebert=not args.no_codebert
    )
    
    # 提取特征
    extractor.extract_all_features(output_dir=output_dir)
    
    print("\n 特征提取完成！")


if __name__ == "__main__":
    main()
