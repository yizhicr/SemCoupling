import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class StaticFunctionFeatureExtractor:
    def __init__(self, project_dir: str, use_codebert: bool = True):
        self.project_dir = Path(project_dir)
        self.use_codebert = use_codebert
        
        print("="*80)
        print(f"第一阶段：静态函数特征提取 - 项目: {self.project_dir.name}")
        print("="*80)
        
        # 加载调用图和元数据
        self._load_static_data()
        
        # 计算特征
        self._compute_features()
    
    def _load_static_data(self):
        """加载静态分析结果"""
        # 函数元数据
        meta_path = self.project_dir / "static_analysis" / "function_metadata.json"
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.function_metadata = json.load(f)
        print(f"函数元数据: {len(self.function_metadata)} 个函数")
        
        self.node_ids = list(self.function_metadata.keys())
    
    def _compute_features(self):
        """计算所有静态特征"""
        print("\n计算静态特征...")
        
        # 1. 语义特征：TF-IDF（CodeBERT可选）
        bodies = [meta.get('body_code', '') for meta in self.function_metadata.values()]
        self.tfidf = TfidfVectorizer(max_features=256).fit_transform(bodies)
        print(f"  TF-IDF 维度: {self.tfidf.shape[1]}")
        
        if self.use_codebert:
            try:
                from data.code_embedder import CodeBERTEmbedder
                self.codebert = CodeBERTEmbedder()
                self.codebert_embeddings = {}
                for node_id, meta in tqdm(self.function_metadata.items(), desc="CodeBERT嵌入"):
                    body = meta.get('body_code', '')
                    if body.strip():
                        emb = self.codebert.embed_function(body).numpy()
                        self.codebert_embeddings[node_id] = emb
                print(f"CodeBERT 嵌入: {len(self.codebert_embeddings)} 个函数")
            except Exception as e:
                print(f"CodeBERT 失败: {e}，使用零向量")
                self.use_codebert = False
        
        # 2. 代码度量特征：行数、参数个数
        self.metrics = {}
        for node_id, meta in self.function_metadata.items():
            loc = meta.get('end_line', 0) - meta.get('start_line', 0)
            args = meta.get('arg_count', 0)
            self.metrics[node_id] = np.array([loc, args], dtype=np.float32)
        
        # 组装每个函数的特征向量
        self.feature_vectors = []
        for node_id in tqdm(self.node_ids, desc="组装特征向量"):
            vec = self._build_feature_vector(node_id)
            self.feature_vectors.append(vec)
        
        self.feature_matrix = np.stack(self.feature_vectors)
        print(f"最终特征维度: {self.feature_matrix.shape[1]}")
    
    def _build_feature_vector(self, node_id: str) -> np.ndarray:
        """构建单个函数的特征向量"""
        features = []
        
        # 语义特征（256维 TF-IDF）
        idx = self.node_ids.index(node_id)
        tfidf_vec = self.tfidf[idx].toarray().flatten()
        features.extend(tfidf_vec)
        
        # CodeBERT（768维）
        if self.use_codebert and node_id in self.codebert_embeddings:
            features.extend(self.codebert_embeddings[node_id])
        else:
            features.extend([0.0] * 768)
        
        # 度量特征（2维）
        features.extend(self.metrics[node_id])
        
        return np.array(features, dtype=np.float32)
    
    def save_features(self, output_dir: str):
        """保存特征矩阵和ID列表"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path / "function_features.npy", self.feature_matrix)
        with open(output_path / "function_ids.json", 'w') as f:
            json.dump(self.node_ids, f, indent=2)
        
        print(f"\n静态特征已保存至: {output_path}")
        print(f"  特征矩阵: {self.feature_matrix.shape}")
        print(f"  函数数量: {len(self.node_ids)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--projects-dir", default="./projects")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-codebert", action="store_true")
    args = parser.parse_args()
    
    project_dir = Path(args.projects_dir) / args.project
    output_dir = args.output_dir or str(project_dir / "features")
    
    extractor = StaticFunctionFeatureExtractor(
        str(project_dir),
        use_codebert=not args.no_codebert
    )
    extractor.save_features(output_dir)


if __name__ == "__main__":
    main()