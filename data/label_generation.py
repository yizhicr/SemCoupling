import os
import sys
import json
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
from datetime import datetime
import networkx as nx
import pickle
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.data_cleaning import TreeSitterParser, get_file_extension


class CommitDiffParser:
    """
    1：Commit 解析与行号提取
    
    从 commits.json 中提取每个 commit 中精确修改的行号范围。
    """
    
    def __init__(self, commits_json_path: str):
        """
        Args:
            commits_json_path: commits.json 文件路径
        """
        self.commits_json_path = commits_json_path
        self.commits = self._load_commits()
        
    def _load_commits(self) -> List[Dict]:
        """加载 commits.json"""
        print(f"正在加载 commits.json...")
        with open(self.commits_json_path, 'r', encoding='utf-8', errors='ignore') as f:
            commits = json.load(f)
        print(f"已加载 {len(commits)} 个 commit")
        return commits
    
    def parse_diff_to_lines(self, diff_text: str) -> Tuple[List[int], List[int]]:
        """
        解析 git diff 文本，提取新增行和删除行的行号
        
        Args:
            diff_text: git diff 文本
            
        Returns:
            (added_lines, deleted_lines) - 新增行号列表和删除行号列表
        """
        added_lines = []
        deleted_lines = []
        
        if not diff_text or diff_text.strip() == '':
            return added_lines, deleted_lines
        
        lines = diff_text.split('\n')
        current_old_line = 0
        current_new_line = 0
        
        # 查找 @@ -old_start,old_count +new_start,new_count @@ 行
        for line in lines:
            if line.startswith('@@'):
                # 解析 hunk header
                match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
                if match:
                    old_start = int(match.group(1))
                    new_start = int(match.group(3))
                    current_old_line = old_start
                    current_new_line = new_start
            elif line.startswith('+') and not line.startswith('+++'):
                # 新增行
                added_lines.append(current_new_line)
                current_new_line += 1
            elif line.startswith('-') and not line.startswith('---'):
                # 删除行
                deleted_lines.append(current_old_line)
                current_old_line += 1
            elif not line.startswith('\\'):
                # 上下文行（未修改）
                current_old_line += 1
                current_new_line += 1
        
        return added_lines, deleted_lines
    
    def extract_commit_details(self) -> Dict[str, Dict]:
        """
        提取所有 commit 的详细修改信息
        
        Returns:
            {
                commit_hash: {
                    "timestamp": str,
                    "author": str,
                    "file_changes": {
                        "relative/path/to/file.py": {
                            "added_lines": [line_nums],
                            "deleted_lines": [line_nums],
                            "all_modified_lines": [line_nums]
                        }
                    }
                }
            }
        """
        print("正在解析所有 commit 的修改行号...")
        commit_details = {}
        
        for commit in tqdm(self.commits, desc="解析 commits"):
            hash_val = commit['hash']
            timestamp = commit.get('committer_date', '')
            author = commit.get('author', {}).get('name', '')
            
            file_changes = {}
            
            for file_mod in commit.get('files_modified', []):
                filename = file_mod.get('new_path') or file_mod.get('filename')
                if not filename:
                    continue
                
                diff = file_mod.get('diff', '')
                added_lines, deleted_lines = self.parse_diff_to_lines(diff)
                
                # 合并所有修改行（去重）
                all_modified = sorted(list(set(added_lines + deleted_lines)))
                
                if all_modified:  # 只记录有实际修改的文件
                    file_changes[filename] = {
                        'added_lines': added_lines,
                        'deleted_lines': deleted_lines,
                        'all_modified_lines': all_modified
                    }
            
            if file_changes:  # 只保存有文件修改的 commit
                commit_details[hash_val] = {
                    'timestamp': timestamp,
                    'author': author,
                    'file_changes': file_changes
                }
        
        print(f"成功解析 {len(commit_details)} 个有效 commit")
        return commit_details


def find_git_root(path: str) -> Optional[str]:
    """
    查找 Git 仓库根目录
    
    Args:
        path: 起始路径
        
    Returns:
        Git 仓库根目录路径，未找到返回 None
    """
    path = Path(path).resolve()
    # 遍历当前路径及所有父目录，找到包含 .git 目录的最近父目录
    for parent in [path] + list(path.parents):
        git_dir = parent / '.git'
        if git_dir.is_dir():  # 确保 .git 是目录而非文件
            return str(parent)
    return None


class HistoricalFileRetriever:
    """
    2：历史文件快照检索
    
    对于给定的 commit 和文件路径，获取该文件在父 commit 中的版本内容。
    """
    
    def __init__(self, repo_path: str):
        """
        Args:
            repo_path: Git 仓库根目录路径（包含 .git 目录）
        """
        self.repo_path = repo_path
        self._error_printed = False  # 用于控制错误信息只打印一次
        
    def get_file_at_commit(self, commit_hash: str, file_path: str) -> Optional[str]:
        """
        获取指定 commit 中的文件内容
        
        Args:
            commit_hash: commit hash
            file_path: 文件相对路径
            
        Returns:
            文件内容字符串，失败返回 None
        """
        try:
            # 将路径中的反斜杠转换为正斜杠，以兼容 Git
            git_path = file_path.replace('\\', '/')
            
            cmd = ['git', 'show', f'{commit_hash}:{git_path}']
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10,
                encoding='utf-8',
                errors='ignore'
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                # 只在前几次失败时打印错误，避免刷屏
                if not self._error_printed:
                    self._error_printed = True
                    print(f"[调试] git show 失败示例:")
                    print(f"  命令: {' '.join(cmd)}")
                    print(f"  错误: {result.stderr.strip()}")
                return None
        except subprocess.TimeoutExpired:
            print(f"获取文件超时 {commit_hash}:{file_path}")
            return None
        except Exception as e:
            print(f"获取文件失败 {commit_hash}:{file_path} - {e}")
            return None
    
    def get_file_at_parent_commit(self, commit_hash: str, file_path: str) -> Optional[str]:
        """
        获取父 commit 中的文件内容（修改前的版本）
        
        Args:
            commit_hash: 当前 commit hash
            file_path: 文件相对路径
            
        Returns:
            父 commit 中的文件内容，失败返回 None
        """
        try:
            # 获取父 commit hash
            cmd = ['git', 'rev-parse', f'{commit_hash}^']
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=5,
                encoding='utf-8',
                errors='ignore'
            )
            
            if result.returncode != 0:
                # 可能是初始 commit，没有父 commit
                if not self._error_printed:
                    self._error_printed = True
                    print(f"[调试] 无法获取父 commit: {result.stderr.strip()}")
                return None
            
            parent_hash = result.stdout.strip()
            return self.get_file_at_commit(parent_hash, file_path)
        except subprocess.TimeoutExpired:
            print(f"获取父 commit 超时 {commit_hash}:{file_path}")
            return None
        except Exception as e:
            print(f"获取父 commit 文件失败 {commit_hash}:{file_path} - {e}")
            return None


class HistoricalFunctionLocator:
    """
    3：函数定位器（在历史代码上解析）
    
    给定历史版本的文件内容和修改行号，找出包含这些行号的所有函数定义。
    """
    
    def __init__(self):
        self.parser_cache = {}  # 缓存不同语言的 parser
    
    def _get_parser(self, lang: str) -> Optional[TreeSitterParser]:
        """获取或创建 parser"""
        if lang not in self.parser_cache:
            try:
                parser = TreeSitterParser()
                self.parser_cache[lang] = parser
            except Exception as e:
                print(f"无法创建 {lang} 的 parser - {e}")
                return None
        return self.parser_cache[lang]
    
    def locate_functions_in_lines(self, source_code: str, modified_lines: List[int], file_path: str) -> List[Dict]:
        """
        在源代码中定位包含修改行号的函数
        
        Args:
            source_code: 历史版本的源代码
            modified_lines: 修改的行号列表
            file_path: 文件路径（用于确定语言）
            
        Returns:
            函数信息列表，每个元素包含：
            {
                'function_name': str,
                'class_name': str or None,
                'start_line': int,
                'end_line': int,
                'body_code': str,
                'matched_lines': [line_nums]  # 该函数覆盖的修改行
            }
        """
        lang = get_file_extension(file_path)
        if not lang:
            return []
        
        parser = self._get_parser(lang)
        if not parser:
            return []
        
        try:
            # 直接调用 extract_functions，无需手动 parse
            functions = parser.extract_functions(source_code, lang)
            
            # 为每个修改行找到所属的函数
            function_map = defaultdict(lambda: {
                'function_name': '',
                'class_name': None,
                'start_line': 0,
                'end_line': 0,
                'body_code': '',
                'matched_lines': []
            })
            
            for line_num in modified_lines:
                for func_info in functions:
                    start_line = func_info['start_line']
                    end_line = func_info['end_line']
                    
                    # 放宽匹配范围，允许上下浮动一行（处理函数声明行等边界情况）
                    if start_line - 1 <= line_num <= end_line + 1:
                        # 使用函数签名作为唯一标识
                        func_key = f"{func_info['file_path']}::{func_info.get('class_name', '')}::{func_info['name']}"
                        
                        func_map = function_map[func_key]
                        func_map['function_name'] = func_info['name']
                        func_map['class_name'] = func_info.get('class_name')
                        func_map['start_line'] = start_line
                        func_map['end_line'] = end_line
                        func_map['body_code'] = func_info.get('body_code', '')
                        if line_num not in func_map['matched_lines']:
                            func_map['matched_lines'].append(line_num)
            
            # 转换为列表
            result = list(function_map.values())
            return result
            
        except Exception as e:
            print(f"函数定位失败 {file_path} - {e}")
            return []


class FunctionIdentityTracker:
    """
    4：函数身份追踪（跨版本映射到当前节点）
    
    将历史 commit 中识别出的函数匹配到当前代码库中的函数节点 ID。
    """
    
    def __init__(
        self, 
        function_metadata: Dict[str, Dict],
        source_code_dir: str
    ):
        """
        Args:
            function_metadata: 当前代码库的函数元数据字典
            source_code_dir: 源代码目录路径
        """
        self.function_metadata = function_metadata
        self.source_code_dir = source_code_dir
        
        # 构建索引以加速匹配
        self._build_indices()
        
        # TF-IDF 向量化工具（用于相似度匹配）
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        self._precompute_tfidf()
    
    def _build_indices(self):
        """构建快速查找索引"""
        # 索引1: (file_path, function_name) -> node_id
        self.file_func_index = {}
        # 索引2: (file_path, class_name, function_name) -> node_id
        self.full_index = {}
        # 索引3: function_name -> [node_ids] （可能有重名函数）
        self.name_index = defaultdict(list)
        
        for node_id, metadata in self.function_metadata.items():
            file_path = metadata.get('file_path', '')
            func_name = metadata.get('name', '')
            class_name = metadata.get('class_name', '')
            
            # 标准化路径：移除 source_code 前缀，统一使用 / 分隔符
            file_path_normalized = self._normalize_file_path(file_path)
            
            key1 = (file_path_normalized, func_name)
            key2 = (file_path_normalized, class_name, func_name)
            
            self.file_func_index[key1] = node_id
            self.full_index[key2] = node_id
            self.name_index[func_name].append(node_id)
        print("\n[调试] 文件-函数索引示例 (前5个):")
        for i, (key, node_id) in enumerate(list(self.file_func_index.items())[:5]):
            print(f"  {i+1}. 路径: '{key[0]}' | 函数名: '{key[1]}'")
        print()
    
    def _normalize_file_path(self, file_path: str) -> str:
        """
        标准化文件路径，移除 source_code 等前缀
        
        Args:
            file_path: 原始文件路径
            
        Returns:
            标准化后的相对路径
        """
        if not file_path:
            return ''
        
        # 统一使用 / 分隔符
        normalized = file_path.replace('\\', '/')
        
        # 移除常见的前缀模式（按优先级）
        prefixes_to_remove = [
            'MaaAssistantArknights/source_code/',
            'MaaAssistantArknights\\source_code\\',# 特定项目前缀
            '/source_code/',
            'source_code/',
            '/projects/',
            'projects/',
        ]
        
        for prefix in prefixes_to_remove:
            prefix_norm = prefix.replace('\\', '/')
            if prefix_norm in normalized:
                # 提取 source_code 之后的部分
                parts = normalized.split(prefix_norm, 1)
                if len(parts) > 1:
                    normalized = parts[1]
                    break
        
        # 移除开头的 /
        normalized = normalized.lstrip('/')
        
        return normalized
    
    def _precompute_tfidf(self):
        """预计算所有函数体的 TF-IDF 向量"""
        if not self.function_metadata:
            return
        
        func_bodies = []
        self.node_ids_ordered = []
        
        for node_id, metadata in self.function_metadata.items():
            body = metadata.get('body_code', '')
            func_bodies.append(body)
            self.node_ids_ordered.append(node_id)
        
        if func_bodies:
            try:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(func_bodies)
            except Exception as e:
                print(f"TF-IDF 预计算失败 - {e}")
                self.tfidf_matrix = None
        else:
            self.tfidf_matrix = None
    
    def match_historical_function(
        self,
        hist_file_path: str,
        hist_func_name: str,
        hist_class_name: Optional[str],
        hist_body_code: str = ''
    ) -> Optional[str]:
        """
        将历史函数映射到当前节点 ID
        
        匹配策略（按优先级）：
        1. 精确路径+类名+函数名匹配
        2. 路径+函数名匹配（忽略类名变化）
        3. 后缀匹配（处理路径不完全一致）
        4. 文件名+函数名匹配（处理文件移动但函数名不变）
        5. 代码相似度匹配（TF-IDF + 路径加权）
        
        Args:
            hist_file_path: 历史文件路径
            hist_func_name: 历史函数名
            hist_class_name: 历史类名（可能为 None）
            hist_body_code: 历史函数体代码（用于相似度匹配）
            
        Returns:
            匹配的当前节点 ID，失败返回 None
        """
        # 标准化历史文件路径（移除 source_code 等前缀）
        hist_file_normalized = self._normalize_file_path(hist_file_path)
        
        # 策略1: 精确匹配（路径 + 类名 + 函数名）
        if hist_class_name:
            key = (hist_file_normalized, hist_class_name, hist_func_name)
            if key in self.full_index:
                return self.full_index[key]
        
        # 策略2: 路径 + 函数名匹配（忽略类名）
        key = (hist_file_normalized, hist_func_name)
        if key in self.file_func_index:
            return self.file_func_index[key]
        
        # 策略2.5: 尝试后缀匹配（处理路径不完全一致的情况）
        # 例如: hist="maths/area.py", current="Python/source_code/maths/area.py"
        for (file_path, func_name), node_id in self.file_func_index.items():
            if func_name == hist_func_name and file_path.endswith(hist_file_normalized):
                return node_id
        
        # 策略2.6: 仅文件名 + 函数名匹配（处理文件移动但函数名不变的情况）
        from pathlib import Path as PathLib
        hist_file_basename = PathLib(hist_file_normalized).name
        for (file_path, func_name), node_id in self.file_func_index.items():
            if func_name == hist_func_name and PathLib(file_path).name == hist_file_basename:
                return node_id
        
        # 策略3: 仅函数名匹配（可能在其他文件中），使用改进的 TF-IDF 相似度
        candidates = self.name_index.get(hist_func_name, [])
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1 and hist_body_code:
            # 多个候选，使用改进的 TF-IDF + 路径相似度选择最佳匹配
            return self._match_by_similarity(
                hist_func_name, 
                hist_body_code, 
                candidates,
                hist_file_path=hist_file_normalized
            )
        
        return None
    
    def _match_by_similarity(
        self, 
        func_name: str, 
        hist_body: str, 
        candidate_node_ids: List[str],
        hist_file_path: str = ''
    ) -> Optional[str]:
        """
        通过 TF-IDF 相似度 + 路径加权匹配函数
        
        Args:
            func_name: 函数名
            hist_body: 历史函数体
            candidate_node_ids: 候选节点 ID 列表
            hist_file_path: 历史文件路径（用于路径相似度计算）
            
        Returns:
            最相似的节点 ID，或 None
        """
        if not self.tfidf_matrix is None and hist_body.strip():
            try:
                from pathlib import Path as PathLib
                
                # 转换历史函数体
                hist_vec = self.tfidf_vectorizer.transform([hist_body])
                
                best_score = -1
                best_node_id = None
                
                for node_id in candidate_node_ids:
                    idx = self.node_ids_ordered.index(node_id)
                    sim = cosine_similarity(hist_vec, self.tfidf_matrix[idx:idx+1])[0][0]
                    
                    # 路径相似度奖励：文件路径相同的候选给予加分
                    current_file = self.function_metadata[node_id].get('file_path', '')
                    if current_file == hist_file_path:
                        sim += 0.2  # 完全相同路径，大幅加分
                    elif PathLib(current_file).name == PathLib(hist_file_path).name:
                        sim += 0.1  # 文件名相同，小幅加分
                    
                    if sim > best_score:
                        best_score = sim
                        best_node_id = node_id
                
                # 降低阈值到 0.5，提高匹配成功率
                if best_score > 0.5:
                    return best_node_id
            except Exception as e:
                print(f"相似度匹配失败 - {e}")
        
        return None


class CoChangePairMiner:
    """
    5：共改对挖掘与严格过滤
    
    从所有 commit 中聚合函数共现关系，应用高置信度阈值，生成最终的正负样本标签。
    """
    
    def __init__(
        self,
        call_graph: nx.DiGraph,
        function_metadata: Dict[str, Dict],
        commit_to_functions: Dict[str, List[str]],
        commits_details: Dict[str, Dict]
    ):
        """
        Args:
            call_graph: 函数调用图
            function_metadata: 函数元数据
            commit_to_functions: {commit_hash: [node_ids]}
            commits_details: commit 详细信息（包含作者、时间戳等）
        """
        self.call_graph = call_graph
        self.function_metadata = function_metadata
        self.commit_to_functions = commit_to_functions
        self.commits_details = commits_details
        
        # 统计信息
        self.total_commits = len(commit_to_functions)
        self.func_modify_count = Counter()  # 每个函数的总修改次数
        
        # 计算每个函数的修改次数
        for node_ids in commit_to_functions.values():
            for node_id in node_ids:
                self.func_modify_count[node_id] += 1
    
    def build_cochange_matrix(self) -> Dict[Tuple[str, str], Dict]:
        """
        构建共改矩阵
        
        Returns:
            {
                (func1_id, func2_id): {
                    'cochange_count': int,
                    'timestamps': [str],
                    'authors': [str],
                    'time_gaps': [float]  # 相邻共改的时间间隔（天）
                }
            }
        """
        print("正在构建共改矩阵...")
        cochange_matrix = defaultdict(lambda: {
            'cochange_count': 0,
            'timestamps': [],
            'authors': [],
            'time_gaps': []
        })
        
        # 按时间排序 commits
        sorted_commits = sorted(
            self.commit_to_functions.items(),
            key=lambda x: self.commits_details.get(x[0], {}).get('timestamp', '')
        )
        
        prev_timestamps = {}  # 记录每对函数上次共改的时间
        
        for commit_hash, node_ids in tqdm(sorted_commits, desc="处理 commits"):
            if len(node_ids) < 2:
                continue
            
            commit_info = self.commits_details.get(commit_hash, {})
            timestamp_str = commit_info.get('timestamp', '')
            author = commit_info.get('author', '')
            
            # 解析时间戳
            try:
                current_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except:
                current_time = None
            
            # 生成所有函数对 C(n, 2)
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    func1 = node_ids[i]
                    func2 = node_ids[j]
                    
                    # 确保有序
                    pair = tuple(sorted([func1, func2]))
                    
                    entry = cochange_matrix[pair]
                    entry['cochange_count'] += 1
                    entry['timestamps'].append(timestamp_str)
                    entry['authors'].append(author)
                    
                    # 计算时间间隔
                    if current_time and pair in prev_timestamps:
                        time_gap = (current_time - prev_timestamps[pair]).days
                        entry['time_gaps'].append(time_gap)
                    
                    if current_time:
                        prev_timestamps[pair] = current_time
        
        print(f"共改矩阵构建完成，共 {len(cochange_matrix)} 个函数对")
        return dict(cochange_matrix)
    
    def filter_positive_pairs(
        self,
        cochange_matrix: Dict[Tuple[str, str], Dict],
        min_cochange_count: int = 3,
        max_distance: int = 3,
        max_modify_ratio: float = 0.2,
        min_unique_authors: int = 2,
        max_avg_time_gap: float = 30.0
    ) -> List[Dict]:
        """
        筛选高质量正样本对
        
        Args:
            cochange_matrix: 共改矩阵
            min_cochange_count: 最小共改次数
            max_distance: 最大调用距离（1=直接调用）
            max_modify_ratio: 函数修改次数占总 commit 数的最大比例
            min_unique_authors: 最少不同作者数
            max_avg_time_gap: 最大平均共改时间间隔（天）
            
        Returns:
            正样本列表
        """
        print("正在筛选正样本对...")
        positive_pairs = []
        
        threshold = max(min_cochange_count, int(self.total_commits * 0.001))
        
        for (func1, func2), stats in tqdm(cochange_matrix.items(), desc="筛选正样本"):
            # 条件1: 共改次数 >= 阈值
            if stats['cochange_count'] < threshold:
                continue
            
            # 条件2: 调用距离 <= 3（直接调用）
            distance = self._calculate_call_distance(func1, func2)
            if distance > max_distance:
                continue
            
            # 条件3: 排除"万能函数"（修改过于频繁）
            modify_threshold = self.total_commits * max_modify_ratio
            if (self.func_modify_count[func1] > modify_threshold or 
                self.func_modify_count[func2] > modify_threshold):
                continue
            
            # 条件4: 至少 2 个不同作者
            unique_authors = len(set(stats['authors']))
            if unique_authors < min_unique_authors:
                continue
            
            # 条件5: 平均时间间隔 < 30 天
            if stats['time_gaps']:
                avg_time_gap = np.mean(stats['time_gaps'])
                if avg_time_gap > max_avg_time_gap:
                    continue
            else:
                avg_time_gap = 0
            
            positive_pairs.append({
                'func1': func1,
                'func2': func2,
                'label': 1,
                'metadata': {
                    'cochange_count': stats['cochange_count'],
                    'distance': distance,
                    'unique_authors': unique_authors,
                    'avg_time_gap': avg_time_gap,
                    'func1_modify_count': self.func_modify_count[func1],
                    'func2_modify_count': self.func_modify_count[func2],
                    'timestamps': stats['timestamps']  # 添加时间戳信息用于时间序列分割
                }
            })
        
        print(f"筛选出 {len(positive_pairs)} 个正样本对")
        return positive_pairs
    
    def _calculate_call_distance(self, func1: str, func2: str) -> int:
        """
        计算两个函数在调用图中的最短距离（BFS）
        
        Returns:
            距离值，不可达返回 -1
        """
        # 检查节点是否存在于图中
        if func1 not in self.call_graph or func2 not in self.call_graph:
            return -1
        
        # 双向 BFS，捕获 NetworkX 异常
        try:
            # 正向：func1 -> func2
            if nx.has_path(self.call_graph, func1, func2):
                dist_forward = nx.shortest_path_length(self.call_graph, func1, func2)
            else:
                dist_forward = float('inf')
            
            # 反向：func2 -> func1
            if nx.has_path(self.call_graph, func2, func1):
                dist_backward = nx.shortest_path_length(self.call_graph, func2, func1)
            else:
                dist_backward = float('inf')
            
            distance = min(dist_forward, dist_backward)
            return int(distance) if distance != float('inf') else -1
        except (nx.NetworkXError, nx.NetworkXNoPath, Exception) as e:
            # 捕获所有 NetworkX 相关异常，避免程序中断
            print(f"警告: 计算调用距离失败 {func1} <-> {func2}: {e}")
            return -1
    
    def sample_negative_pairs(
        self,
        positive_pairs: List[Dict],
        negative_ratio: float = 2.5
    ) -> List[Dict]:
        """
        采样负样本对
        
        策略：
        1. Hard Negative: 调用图中直接调用但从未共改的函数对
        2. Intra-file Negative: 同一文件内无调用关系且从未共改的函数对
        
        Args:
            positive_pairs: 正样本列表
            negative_ratio: 负样本与正样本的比例
            
        Returns:
            负样本列表
        """
        print("正在采样负样本对...")
        
        # 构建正样本集合（快速查找）
        positive_set = set()
        for pair in positive_pairs:
            pair_key = tuple(sorted([pair['func1'], pair['func2']]))
            positive_set.add(pair_key)
        
        negative_pairs = []
        target_count = int(len(positive_pairs) * negative_ratio)
        
        # 策略1: Hard Negative - 直接调用但未共改
        hard_negatives = self._sample_hard_negatives(positive_set)
        negative_pairs.extend(hard_negatives[:target_count // 2])
        
        # 策略2: Intra-file Negative - 同文件无调用关系
        if len(negative_pairs) < target_count:
            remaining = target_count - len(negative_pairs)
            intra_file_negatives = self._sample_intra_file_negatives(positive_set, remaining)
            negative_pairs.extend(intra_file_negatives)
        
        print(f"采样出 {len(negative_pairs)} 个负样本对")
        return negative_pairs
    
    def _sample_hard_negatives(self, positive_set: Set[Tuple[str, str]]) -> List[Dict]:
        """采样 Hard Negative：直接调用但未共改的函数对"""
        hard_negatives = []
        
        for edge in tqdm(self.call_graph.edges(), desc="采样 Hard Negative"):
            func1, func2 = edge
            pair_key = tuple(sorted([func1, func2]))
            
            if pair_key not in positive_set:
                hard_negatives.append({
                    'func1': func1,
                    'func2': func2,
                    'label': 0,
                    'metadata': {
                        'type': 'hard_negative',
                        'distance': 1,
                        'cochange_count': 0
                    }
                })
        
        return hard_negatives
    
    def _sample_intra_file_negatives(
        self, 
        positive_set: Set[Tuple[str, str]],
        count: int
    ) -> List[Dict]:
        """采样 Intra-file Negative：同文件无调用关系的函数对"""
        # 按文件分组函数
        file_to_funcs = defaultdict(list)
        for node_id, metadata in self.function_metadata.items():
            file_path = metadata.get('file_path', '')
            file_to_funcs[file_path].append(node_id)
        
        intra_file_negatives = []
        
        for file_path, funcs in file_to_funcs.items():
            if len(funcs) < 2:
                continue
            
            # 随机采样同文件函数对
            import random
            max_attempts = count * 2
            attempts = 0
            
            while len(intra_file_negatives) < count and attempts < max_attempts:
                func1, func2 = random.sample(funcs, 2)
                pair_key = tuple(sorted([func1, func2]))
                
                # 检查：不在正样本中、没有调用关系
                if (pair_key not in positive_set and 
                    not self.call_graph.has_edge(func1, func2) and
                    not self.call_graph.has_edge(func2, func1)):
                    
                    intra_file_negatives.append({
                        'func1': func1,
                        'func2': func2,
                        'label': 0,
                        'metadata': {
                            'type': 'intra_file_negative',
                            'file_path': file_path,
                            'cochange_count': 0
                        }
                    })
                
                attempts += 1
            
            if len(intra_file_negatives) >= count:
                break
        
        return intra_file_negatives[:count]
    
    def generate_static_samples(
        self,
        max_distance: int = 2,
        negative_ratio: float = 2.5,
        include_intra_file: bool = True
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        纯静态模式：基于调用图生成正负样本，不依赖 commit 历史
        
        Args:
            max_distance: 正样本允许的最大调用距离（默认2）
            negative_ratio: 负样本与正样本的比例
            include_intra_file: 是否包含同文件负样本
            
        Returns:
            (positive_pairs, negative_pairs)
        """
        print("正在基于调用图生成静态样本...")
        
        # ==================== 生成正样本 ====================
        positive_pairs = []
        positive_set = set()
        
        # 遍历图中所有可能的函数对，计算距离
        nodes = list(self.call_graph.nodes())
        for i, func1 in enumerate(tqdm(nodes, desc="生成正样本")):
            for func2 in nodes[i+1:]:
                # 计算调用距离
                distance = self._calculate_call_distance(func1, func2)
                if distance != -1 and distance <= max_distance:
                    pair_key = tuple(sorted([func1, func2]))
                    if pair_key in positive_set:
                        continue
                    positive_set.add(pair_key)
                    positive_pairs.append({
                        'func1': func1,
                        'func2': func2,
                        'label': 1,
                        'metadata': {
                            'cochange_count': 0,
                            'distance': distance,
                            'unique_authors': 0,
                            'avg_time_gap': 0.0,
                            'func1_modify_count': 0,
                            'func2_modify_count': 0,
                            'timestamps': []      # 静态模式无时间戳
                        }
                    })
        
        print(f"  生成 {len(positive_pairs)} 个正样本（距离 ≤ {max_distance}）")
        
        # ==================== 生成负样本 ====================
        negative_pairs = []
        negative_set = set()
        target_neg = int(len(positive_pairs) * negative_ratio)
        
        # 策略1: Hard Negative（直接调用但未纳入正样本？实际上直接调用已全部成为正样本）
        # 静态模式下，直接调用都算正样本，所以跳过 Hard Negative
        
        # 策略2: 无调用关系的随机函数对
        import random
        random.seed(42)
        
        # 生成候选负样本池
        candidate_pairs = []
        nodes_list = list(self.call_graph.nodes())
        # 随机采样，避免全量遍历
        max_attempts = target_neg * 10
        attempts = 0
        while len(negative_pairs) < target_neg and attempts < max_attempts:
            func1, func2 = random.sample(nodes_list, 2)
            pair_key = tuple(sorted([func1, func2]))
            attempts += 1
            
            if pair_key in positive_set or pair_key in negative_set:
                continue
            
            # 确保无调用关系（距离为 -1）
            if self._calculate_call_distance(func1, func2) != -1:
                continue
            
            negative_set.add(pair_key)
            negative_pairs.append({
                'func1': func1,
                'func2': func2,
                'label': 0,
                'metadata': {
                    'type': 'random_negative',
                    'distance': -1,
                    'cochange_count': 0
                }
            })
        
        # 策略3: 同文件负样本（如果还没够）
        if include_intra_file and len(negative_pairs) < target_neg:
            remaining = target_neg - len(negative_pairs)
            intra_neg = self._sample_intra_file_negatives(positive_set, remaining)
            for pair in intra_neg:
                pair_key = tuple(sorted([pair['func1'], pair['func2']]))
                if pair_key not in negative_set:
                    negative_set.add(pair_key)
                    negative_pairs.append(pair)
        
        print(f"  生成 {len(negative_pairs)} 个负样本")
        return positive_pairs, negative_pairs
    
    def temporal_split(
        self,
        all_pairs: List[Dict],
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        按时间序列分割数据集（正样本）+ 随机分割（负样本）
        
        Args:
            all_pairs: 所有样本对
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            
        Returns:
            (train_set, val_set, test_set)
        """
        import random
        random.seed(42)  # 固定随机种子确保可重现性
        
        # 分离正负样本
        positive_pairs = [p for p in all_pairs if p['label'] == 1]
        negative_pairs = [p for p in all_pairs if p['label'] == 0]
        
        print(f"  正样本总数: {len(positive_pairs)}, 负样本总数: {len(negative_pairs)}")
        
        # ==================== 正样本：按时间序列分割 ====================
        def get_earliest_timestamp(pair):
            timestamps = pair.get('metadata', {}).get('timestamps', [])
            if timestamps:
                return min(timestamps)
            return ''
        
        sorted_positive = sorted(positive_pairs, key=get_earliest_timestamp)
        
        n_pos = len(sorted_positive)
        pos_train_end = int(n_pos * train_ratio)
        pos_val_end = int(n_pos * (train_ratio + val_ratio))
        
        pos_train = sorted_positive[:pos_train_end]
        pos_val = sorted_positive[pos_train_end:pos_val_end]
        pos_test = sorted_positive[pos_val_end:]
        
        print(f"  正样本分割: 训练集={len(pos_train)}, 验证集={len(pos_val)}, 测试集={len(pos_test)}")
        
        # ==================== 负样本：按比例随机分割 ====================
        random.shuffle(negative_pairs)
        
        n_neg = len(negative_pairs)
        neg_train_end = int(n_neg * train_ratio)
        neg_val_end = int(n_neg * (train_ratio + val_ratio))
        
        neg_train = negative_pairs[:neg_train_end]
        neg_val = negative_pairs[neg_train_end:neg_val_end]
        neg_test = negative_pairs[neg_val_end:]
        
        print(f"  负样本分割: 训练集={len(neg_train)}, 验证集={len(neg_val)}, 测试集={len(neg_test)}")
        
        # ==================== 合并正负样本 ====================
        train_set = pos_train + neg_train
        val_set = pos_val + neg_val
        test_set = pos_test + neg_test
        
        # 打乱每个集合的顺序
        random.shuffle(train_set)
        random.shuffle(val_set)
        random.shuffle(test_set)
        
        print(f"数据集分割: 训练集={len(train_set)}, 验证集={len(val_set)}, 测试集={len(test_set)}")
        return train_set, val_set, test_set


def run_label_generation(
    project_name: str,
    projects_dir: str = "./projects",
    min_cochange_count: int = 3,
    negative_ratio: float = 2.5,
    max_commits: Optional[int] = None,
    mode: str = "hybrid"       # 新增：hybrid 或 static
):
    """
    运行完整的标签生成流程
    
    Args:
        project_name: 项目名称
        projects_dir: 项目目录
        min_cochange_count: 最小共改次数阈值
        negative_ratio: 负样本比例
        max_commits: 最大处理的commit数量（None表示不限制）
        mode: 标签生成模式（hybrid=使用commit历史，static=仅调用图）
        
    Returns:
        生成的标签文件路径
    """
    project_dir = Path(projects_dir) / project_name
    source_code_dir = project_dir / "source_code"
    commits_json_path = project_dir / "commits.json"
    
    # 尝试多个可能的调用图路径
    call_graph_path = project_dir / "call_graph.pkl"
    if not call_graph_path.exists():
        # 尝试 static_analysis 子目录
        call_graph_path = project_dir / "static_analysis" / "call_graph.pkl"
    
    output_dir = project_dir / "labels"
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    mode_str = "混合模式（使用commit历史）" if mode == "hybrid" else "静态模式（仅调用图）"
    print(f"第三阶段：标签生成 - 项目: {project_name} [{mode_str}]")
    print("="*80)
    
    # ==================== 步骤 1: 加载基础数据 ====================
    print("\n[步骤 1/6] 加载基础数据...")
    
    # 加载调用图（两种模式都需要）
    if not call_graph_path.exists():
        raise FileNotFoundError(f"call_graph.pkl 不存在: {call_graph_path}")
    
    with open(call_graph_path, 'rb') as f:
        call_graph = pickle.load(f)
    
    # 验证节点 ID 格式
    print("\n  验证节点 ID 格式...")
    sample_nodes = list(call_graph.nodes())[:5]
    print(f"  示例节点 ID:")
    for node_id in sample_nodes:
        parts = node_id.split('::')
        print(f"    - {node_id}")
        if len(parts) != 3:
            print(f"      ⚠ 警告: 节点 ID 格式不正确 (期望3段，实际{len(parts)}段)")
    
    # 加载函数元数据
    function_metadata_path = project_dir / "static_analysis" / "function_metadata.json"
    if function_metadata_path.exists():
        with open(function_metadata_path, 'r', encoding='utf-8') as f:
            function_metadata = json.load(f)
    else:
        # 从调用图节点属性重建
        function_metadata = {}
        for node_id in call_graph.nodes():
            attrs = call_graph.nodes[node_id]
            function_metadata[node_id] = {
                'name': attrs.get('name', node_id.split('::')[-1]),
                'file_path': attrs.get('file_path', ''),
                'class_name': attrs.get('class_name', ''),
                'body_code': attrs.get('body_code', ''),
                'start_line': attrs.get('start_line', 0),
                'end_line': attrs.get('end_line', 0)
            }
    
    print(f"加载完成: {len(call_graph.nodes())} 函数节点")
    
    # ==================== 根据模式分支执行 ====================
    if mode == "static":
        # ========== 纯静态模式 ==========
        print("\n[静态模式] 基于调用图生成样本，不使用 commit 历史")
        
        # 创建 miner 实例（不需要 commits_details）
        miner = CoChangePairMiner(
            call_graph=call_graph,
            function_metadata=function_metadata,
            commit_to_functions={},      # 空字典
            commits_details={}           # 空字典
        )
        
        # 生成静态样本
        positive_pairs, negative_pairs = miner.generate_static_samples(
            max_distance=2,              # 可配置
            negative_ratio=negative_ratio,
            include_intra_file=True
        )
        
        print(f"\n筛选出 {len(positive_pairs)} 个高质量正样本对")
        print(f"采样出 {len(negative_pairs)} 个负样本对")
        
        # 合并样本
        all_pairs = positive_pairs + negative_pairs
        
        # 由于没有时间戳，使用随机分割
        import random
        random.seed(42)
        random.shuffle(all_pairs)
        n = len(all_pairs)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        train_set = all_pairs[:train_end]
        val_set = all_pairs[train_end:val_end]
        test_set = all_pairs[val_end:]
        
        print(f"\n数据集分割: 训练集={len(train_set)}, 验证集={len(val_set)}, 测试集={len(test_set)}")
        
    else:
        # ========== 混合模式（原流程） ==========
        print("\n[混合模式] 使用 commit 历史生成样本")
        
        # 加载 commits.json
        if not commits_json_path.exists():
            raise FileNotFoundError(f"commits.json 不存在: {commits_json_path}")
        
        commit_parser = CommitDiffParser(str(commits_json_path))
        commits_details = commit_parser.extract_commit_details()
        
        # 如果指定了max_commits，限制commit数量
        if max_commits is not None and max_commits > 0:
            print(f"\n  限制处理前 {max_commits} 个 commits...")
            # 按时间排序，取最近的max_commits个
            sorted_commits = sorted(
                commits_details.items(),
                key=lambda x: x[1].get('timestamp', ''),
                reverse=True  # 最近的在前
            )
            commits_details = dict(sorted_commits[:max_commits])
            print(f"  实际处理 {len(commits_details)} 个 commits")
        
        # ==================== 步骤 2: 历史函数提取与映射（简化版） ====================
        print("\n[步骤 2/6] 历史函数提取与映射（基于函数名直接匹配）...")
        
        repo_root = find_git_root(str(source_code_dir))
        if not repo_root:
            raise RuntimeError(f"未找到 Git 仓库根目录，请检查 {source_code_dir}")
        print(f"  Git 仓库根目录: {repo_root}")
        
        # 预过滤：筛选至少修改2个源代码文件的 commit
        print("\n[预过滤] 筛选至少修改2个源代码文件的 commit...")
        filtered_commits_details = {}
        source_extensions = {'.cpp', '.c', '.h', '.hpp', '.cxx', '.cc', '.cs', '.java', '.py', '.js', '.ts'}
        
        for commit_hash, details in tqdm(commits_details.items(), desc="预过滤 commits"):
            source_file_count = 0
            for file_path in details['file_changes'].keys():
                ext = Path(file_path).suffix.lower()
                if ext in source_extensions:
                    source_file_count += 1
                    if source_file_count >= 2:
                        filtered_commits_details[commit_hash] = details
                        break
        
        print(f"  原始 commits: {len(commits_details)}")
        print(f"  过滤后 commits: {len(filtered_commits_details)}")
        commits_details = filtered_commits_details
        
        retriever = HistoricalFileRetriever(repo_root)
        tracker = FunctionIdentityTracker(function_metadata, str(source_code_dir))
        
        # 创建一个简易的函数提取器缓存
        parser_cache = {}
        def get_parser(lang):
            if lang not in parser_cache:
                parser_cache[lang] = TreeSitterParser()
            return parser_cache[lang]
        
        commit_to_functions = {}
        unmapped_count = 0
        total_functions_found = 0
        processed_files = 0
        files_with_functions = 0
        
        for commit_hash, details in tqdm(commits_details.items(), desc="映射历史函数"):
            mapped_node_ids = set()
            
            for file_path, changes in details['file_changes'].items():
                # 获取历史文件内容（父 commit 或当前 commit）
                content = retriever.get_file_at_parent_commit(commit_hash, file_path)
                if not content:
                    content = retriever.get_file_at_commit(commit_hash, file_path)
                if not content:
                    continue
                
                lang = get_file_extension(file_path)
                if not lang:
                    continue
                
                parser = get_parser(lang)
                try:
                    functions = parser.extract_functions(content, lang)
                except Exception:
                    continue
                
                if not functions:
                    continue
                
                processed_files += 1
                if functions:
                    files_with_functions += 1
                
                # 对该文件中的每个函数，尝试匹配到当前节点ID
                for func_info in functions:
                    node_id = tracker.match_historical_function(
                        hist_file_path=file_path,
                        hist_func_name=func_info['name'],
                        hist_class_name=func_info.get('class_name'),
                        hist_body_code=''   # 不依赖 body_code，主要靠路径+函数名
                    )
                    if node_id:
                        mapped_node_ids.add(node_id)
                        total_functions_found += 1
                    else:
                        unmapped_count += 1
            
            if mapped_node_ids:
                commit_to_functions[commit_hash] = list(mapped_node_ids)
        
        mapping_coverage = total_functions_found / (total_functions_found + unmapped_count) if (total_functions_found + unmapped_count) > 0 else 0
        print(f"映射完成: {len(commit_to_functions)} commits 包含函数, "
              f"函数匹配覆盖率={mapping_coverage:.2%}, 未映射函数数={unmapped_count}")
        print(f"  处理文件数: {processed_files}, 其中包含函数的文件数: {files_with_functions}")
        
        # ==================== 步骤 3: 构建共改矩阵 ====================
        print("\n[步骤 3/6] 构建共改矩阵...")
        
        miner = CoChangePairMiner(
            call_graph=call_graph,
            function_metadata=function_metadata,
            commit_to_functions=commit_to_functions,
            commits_details=commits_details
        )
        
        cochange_matrix = miner.build_cochange_matrix()
        
        # ==================== 步骤 4: 筛选正样本对 ====================
        print("\n[步骤 4/6] 筛选正样本对...")
        
        positive_pairs = miner.filter_positive_pairs(
            cochange_matrix=cochange_matrix,
            min_cochange_count=min_cochange_count,
            max_distance=3,
            max_modify_ratio=0.1,
            min_unique_authors=1,
            max_avg_time_gap=30.0
        )
        
        print(f"筛选出 {len(positive_pairs)} 个高质量正样本对")
        
        # ==================== 步骤 5: 采样负样本对 ====================
        print("\n[步骤 5/6] 采样负样本对...")
        
        negative_pairs = miner.sample_negative_pairs(
            positive_pairs=positive_pairs,
            negative_ratio=negative_ratio
        )
        
        print(f"采样出 {len(negative_pairs)} 个负样本对")
        
        # ==================== 步骤 6: 保存结果 ====================
        print("\n[步骤 6/6] 保存标签数据...")
        
        # 合并所有样本
        all_pairs = positive_pairs + negative_pairs
        
        # 按时间序列分割数据集
        train_set, val_set, test_set = miner.temporal_split(
            all_pairs=all_pairs,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
    
    # ==================== 步骤 6: 保存结果（两种模式共用）====================
    print("\n[步骤 6/6] 保存标签数据...")
    
    # 保存训练标签
    training_labels_path = output_dir / "training_labels_v2.json"
    with open(training_labels_path, 'w', encoding='utf-8') as f:
        json.dump({
            'train': train_set,
            'val': val_set,
            'test': test_set,
            'statistics': {
                'total_pairs': len(all_pairs),
                'positive_pairs': len(positive_pairs),
                'negative_pairs': len(negative_pairs),
                'train_size': len(train_set),
                'val_size': len(val_set),
                'test_size': len(test_set),
                'min_cochange_threshold': min_cochange_count if mode == "hybrid" else 0,
                'negative_ratio': negative_ratio,
                'generation_mode': mode  # 记录生成模式
            }
        }, f, indent=2, ensure_ascii=False)
    
    # 保存正负样本对（用于后续分析）
    positive_pairs_path = output_dir / "positive_pairs.json"
    with open(positive_pairs_path, 'w', encoding='utf-8') as f:
        json.dump(positive_pairs, f, indent=2, ensure_ascii=False)
    
    negative_pairs_path = output_dir / "negative_pairs.json"
    with open(negative_pairs_path, 'w', encoding='utf-8') as f:
        json.dump(negative_pairs, f, indent=2, ensure_ascii=False)
    
    # 保存 commit 到函数的映射关系（仅混合模式）
    if mode == "hybrid":
        commit_mapping_path = output_dir / "commit_to_functions.json"
        with open(commit_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(commit_to_functions, f, indent=2, ensure_ascii=False)
    
    print(f"\n 标签生成完成！")
    print(f"  训练集: {len(train_set)} 个样本")
    print(f"  验证集: {len(val_set)} 个样本")
    print(f"  测试集: {len(test_set)} 个样本")
    print(f"  输出目录: {output_dir}")
    
    return str(output_dir)


def main():
    """主函数 - 命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="第三阶段：标签生成 - 基于历史提交挖掘函数共改关系")
    parser.add_argument("--project", type=str, required=True, help="项目名称")
    parser.add_argument("--projects-dir", type=str, default="./projects", help="项目根目录")
    parser.add_argument("--min-cochange", type=int, default=3, help="最小共改次数阈值（默认3）")
    parser.add_argument("--negative-ratio", type=float, default=2.5, help="负样本与正样本比例（默认2.5）")
    parser.add_argument("--max-commits", type=int, default=None, help="最大处理的commit数量（None表示不限制）")
    parser.add_argument("--mode", type=str, default="hybrid", 
                        choices=["hybrid", "static"],
                        help="标签生成模式：hybrid(使用commit历史) 或 static(仅调用图)")
    
    args = parser.parse_args()
    
    try:
        run_label_generation(
            project_name=args.project,
            projects_dir=args.projects_dir,
            min_cochange_count=args.min_cochange,
            negative_ratio=args.negative_ratio,
            max_commits=args.max_commits,
            mode=args.mode
        )
        print("\n 标签生成流程执行成功！")
    except Exception as e:
        print(f"\n 标签生成失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
