import sys
from pathlib import Path

# 添加项目根目录到路径，确保可以导入 data 包中的模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 修正导入路径：从 data.data_cleaning 导入
from data.data_cleaning import TreeSitterParser, get_file_extension


def example_usage():
    """使用示例"""
    print("="*60)
    print("静态代码分析模块")
    print("="*60)
    
    print("运行完整的静态代码分析")
    print("-" * 60)
    
    # 假设我们要分析的项目路径，可以根据实际情况修改
    source_dir = "./projects/LunaTranslator/source_code"
    projects_dir = "./projects"
    project_name = "LunaTranslator"
    
    # 检查目录是否存在
    if not Path(source_dir).exists():
        print(f"目录 {source_dir} 不存在")
        return
    
    try:
        # 运行静态分析
        call_graph, function_metadata = run_static_analysis(
            source_code_dir=source_dir,
            project_name=project_name,
            projects_dir=projects_dir
        )
        
        print("分析完成:")
        print(f"- 调用图节点数: {call_graph.number_of_nodes()}")
        print(f"- 调用图边数: {call_graph.number_of_edges()}")
        print(f"- 函数元数据数量: {len(function_metadata)}")
        
        # 示例2: 查询特定函数的信息
        print("查询特定函数的元数据")
        print("-" * 60)
        
        if function_metadata:
            first_node_id = list(function_metadata.keys())[0]
            metadata = function_metadata[first_node_id]
            
            print(f"函数名: {metadata.name}")
            print(f"文件路径: {metadata.file_path}")
            print(f"行号范围: {metadata.start_line} - {metadata.end_line}")
            print(f"语言: {metadata.language}")
            print(f"参数列表: {metadata.parameters}")
            print(f"所属类: {metadata.class_name or 'N/A'}")
            print(f"模块路径: {metadata.module_path}")
            print(f"参数数量: {metadata.arg_count}")
            body_preview = metadata.body_code[:100].replace('\n', ' ') if metadata.body_code else ""
            print(f"函数体代码（前100字符）: {body_preview}...")
        
        # 示例3: 查询调用关系
        print("查询函数的调用关系")
        print("-" * 60)
        
        if function_metadata:
            sample_node = list(function_metadata.keys())[0]
            
            successors = list(call_graph.successors(sample_node))
            print(f"\n函数 '{sample_node}' 调用了:")
            for succ in successors[:5]:
                print(f"  -> {succ}")
            if len(successors) > 5:
                print(f"  ... 还有 {len(successors) - 5} 个")
            
            predecessors = list(call_graph.predecessors(sample_node))
            print(f"\n调用 '{sample_node}' 的函数:")
            for pred in predecessors[:5]:
                print(f"  <- {pred}")
            if len(predecessors) > 5:
                print(f"  ... 还有 {len(predecessors) - 5} 个")
        
        # 示例4: 加载已保存的结果
        print("加载已保存的分析结果")
        print("-" * 60)
        
        analyzer = StaticCodeAnalyzer(source_dir, projects_dir=projects_dir)
        
        output_dir = Path(projects_dir) / project_name / "static_analysis"
        call_graph_path = output_dir / "call_graph.pkl"
        if call_graph_path.exists():
            loaded_graph = analyzer.load_call_graph(str(call_graph_path))
            print(f"已加载调用图: {loaded_graph.number_of_nodes()} 个节点, "
                  f"{loaded_graph.number_of_edges()} 条边")
        
        metadata_path = output_dir / "function_metadata.json"
        if metadata_path.exists():
            loaded_metadata = analyzer.load_function_metadata(str(metadata_path))
            print(f"已加载函数元数据: {len(loaded_metadata)} 个函数")
        
        # 示例5: 获取统计信息
        print("获取分析统计信息")
        print("-" * 60)
        
        stats = analyzer.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    example_usage()
"""
静态代码分析模块
目标：为每个函数建立精确的元数据档案，并构建可靠的有向调用图
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pickle
import networkx as nx
from tqdm import tqdm
import json
import ast


class FunctionMetadata:
    """函数元数据类"""
    
    def __init__(self, name: str, file_path: str, start_line: int, end_line: int,
                 language: str, parameters: List[str] = None, body_code: str = "",
                 class_name: Optional[str] = None, module_path: Optional[str] = None):
        self.name = name
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.language = language
        self.parameters = parameters or []
        self.body_code = body_code
        self.class_name = class_name
        self.module_path = module_path
        self.arg_count = len(self.parameters)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'name': self.name,
            'file_path': self.file_path,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'language': self.language,
            'parameters': self.parameters,
            'body_code': self.body_code,
            'class_name': self.class_name,
            'module_path': self.module_path,
            'arg_count': self.arg_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FunctionMetadata':
        """从字典创建实例"""
        return cls(
            name=data['name'],
            file_path=data['file_path'],
            start_line=data['start_line'],
            end_line=data['end_line'],
            language=data['language'],
            parameters=data.get('parameters', []),
            body_code=data.get('body_code', ''),
            class_name=data.get('class_name'),
            module_path=data.get('module_path')
        )


class StaticCodeAnalyzer:
    """
    静态代码分析器
    
    功能：
    1. 遍历源代码文件，使用 tree-sitter 解析 AST
    2. 提取所有函数定义及其元数据
    3. 构建有向调用图（caller -> callee）
    4. 为图中每个节点附加丰富的属性
    """
    
    def __init__(self, source_code_dir: str, projects_dir: str = "./projects"):
        """
        初始化静态代码分析器
        
        Args:
            source_code_dir: 源代码目录路径
            projects_dir: 项目根目录（用于路径计算）
        """
        self.source_code_dir = Path(source_code_dir)
        self.projects_dir = Path(projects_dir)
        self.parser = TreeSitterParser()
        
        # 存储所有函数的元数据：node_id -> FunctionMetadata
        self.function_metadata: Dict[str, FunctionMetadata] = {}
        
        # 调用图（有向图）
        self.call_graph: nx.DiGraph = nx.DiGraph()
        
        # 支持的文件扩展名
        self.supported_extensions = {'.py', '.java', '.js', '.ts', '.jsx', '.tsx', 
                                     '.c', '.cpp', '.cxx', '.h', '.hpp'}
    
    def analyze_all_files(self) -> Tuple[nx.DiGraph, Dict[str, FunctionMetadata]]:
        """
        分析所有源代码文件，构建调用图和函数元数据
        
        Returns:
            Tuple[nx.DiGraph, Dict[str, FunctionMetadata]]: 调用图和函数元数据字典
        """
        print(f"开始静态代码分析: {self.source_code_dir}")
        
        # 第一步：收集所有需要分析的源文件
        source_files = self._collect_source_files()
        print(f"找到 {len(source_files)} 个源代码文件")
        
        # 第二步：遍历每个文件，提取函数定义和调用关系
        for file_path in tqdm(source_files, desc="分析源文件"):
            try:
                self._analyze_single_file(file_path)
            except Exception as e:
                print(f"分析文件 {file_path} 时出错: {e}")
                continue
        
        print(f"\n静态代码分析完成")
        print(f"- 提取函数总数: {len(self.function_metadata)}")
        print(f"- 调用图节点数: {self.call_graph.number_of_nodes()}")
        print(f"- 调用图边数: {self.call_graph.number_of_edges()}")
        
        return self.call_graph, self.function_metadata
    
    def _collect_source_files(self) -> List[Path]:
        """
        递归收集所有支持的源代码文件
        
        Returns:
            List[Path]: 源代码文件路径列表
        """
        source_files = []
        
        for root, dirs, files in os.walk(self.source_code_dir):
            # 跳过隐藏目录和虚拟环境
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__' 
                      and d != 'node_modules' and d != '.venv']
            
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in self.supported_extensions:
                    source_files.append(file_path)
        
        return source_files
    
    def _analyze_single_file(self, file_path: Path):
        """
        分析单个文件，提取函数定义和调用关系
        
        Args:
            file_path: 文件路径
        """
        # 确定语言类型
        language = get_file_extension(str(file_path))
        if not language:
            return
        
        # 读取文件内容
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
        except Exception as e:
            print(f"无法读取文件 {file_path}: {e}")
            return
        
        if not code.strip():
            return
        
        # 计算相对路径（相对于 projects 目录）
        try:
            relative_path = str(file_path.relative_to(self.projects_dir))
        except ValueError:
            # 如果无法计算相对路径，使用绝对路径
            relative_path = str(file_path)
        
        # 提取模块路径（去掉文件扩展名的路径）
        module_path = str(Path(relative_path).with_suffix(''))
        
        # 第一步：提取函数定义
        functions = self.parser.extract_functions(code, language)
        
        # 第二步：提取类信息（用于确定函数所属的类）
        classes_info = self._extract_classes(code, language)
        
        # 第三步：解析导入语句，构建导入映射表
        import_map = self._parse_imports(code, language, relative_path)
        
        # 第四步：为每个函数创建元数据并添加到图中
        for func in functions:
            func_name = func['name']
            start_line = func['start_line']
            end_line = func['end_line']
            
            # 确定函数所属的类
            class_name = self._find_enclosing_class(start_line, classes_info)
            
            # 统一节点ID格式：{relative_path}::{class_name or ''}::{func_name}
            # 即使没有类名，也保留空字符串占位符，保证 split('::') 后长度为3
            node_id = f"{relative_path}::{class_name or ''}::{func_name}"
            
            # 提取函数体代码
            body_code = self._extract_function_body(code, start_line, end_line)
            
            # 创建函数元数据
            metadata = FunctionMetadata(
                name=func_name,
                file_path=relative_path,
                start_line=start_line,
                end_line=end_line,
                language=language,
                parameters=func.get('parameters', []),
                body_code=body_code,
                class_name=class_name,
                module_path=module_path
            )
            
            # 存储元数据
            self.function_metadata[node_id] = metadata
            
            # 添加节点到图中，附带属性
            self.call_graph.add_node(node_id, 
                                    name=func_name,
                                    file_path=relative_path,
                                    start_line=start_line,
                                    end_line=end_line,
                                    language=language,
                                    class_name=class_name,
                                    module_path=module_path,
                                    arg_count=len(func.get('parameters', [])),
                                    body_code=body_code)
        
        # 第五步：提取函数调用关系
        calls = self.parser.extract_calls(code, language)
        
        # 第六步：为每个调用建立边（caller -> callee）
        for call in calls:
            called_func_name = call['function']
            call_line = call['line_number']
            
            # 找到调用者函数（包含该行号的函数）
            caller_node_id = self._find_function_at_line(call_line, functions, relative_path, classes_info)
            
            if not caller_node_id:
                continue
            
            # 尝试解析被调用函数的节点ID（使用导入映射表）
            callee_node_id = self._resolve_callee_node_id(
                called_func_name, relative_path, classes_info, call_line, import_map
            )
            
            if callee_node_id:
                # 添加有向边：caller -> callee
                if not self.call_graph.has_edge(caller_node_id, callee_node_id):
                    self.call_graph.add_edge(caller_node_id, callee_node_id,
                                            call_line=call_line)
    
    def _parse_imports(self, code: str, language: str, file_path: str) -> Dict[str, str]:
        """
        解析文件的导入语句，构建导入映射表
        
        Args:
            code: 源代码
            language: 编程语言
            file_path: 文件路径
            
        Returns:
            Dict[str, str]: 导入映射表 {alias -> full_module_path}
        """
        import_map = {}
        
        try:
            if language == 'python':
                import_map.update(self._parse_python_imports(code))
            elif language == 'java':
                import_map.update(self._parse_java_imports(code))
            elif language in ['javascript', 'typescript']:
                import_map.update(self._parse_js_imports(code))
            elif language in ['c', 'cpp']:
                import_map.update(self._parse_c_imports(code))
        except Exception as e:
            # 如果解析失败，返回已收集的部分结果
            print(f"解析 {language} 导入时出错: {e}")
        
        return import_map
    
    def _parse_python_imports(self, code: str) -> Dict[str, str]:
        """
        解析 Python 导入语句
        
        Args:
            code: Python 源代码
            
        Returns:
            Dict[str, str]: 导入映射表 {alias -> full_module_path}
        """
        import_map = {}
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    # 处理 import xxx 或 import xxx as yyy
                    for alias in node.names:
                        name = alias.name
                        asname = alias.asname or name
                        import_map[asname] = name
                elif isinstance(node, ast.ImportFrom):
                    # 处理 from xxx import yyy 或 from xxx import yyy as zzz
                    module = node.module or ''
                    for alias in node.names:
                        name = alias.name
                        asname = alias.asname or name
                        # 记录为 module.name 的形式
                        full_path = f"{module}.{name}" if module else name
                        import_map[asname] = full_path
        except SyntaxError:
            pass
        
        return import_map
    
    def _parse_java_imports(self, code: str) -> Dict[str, str]:
        """
        解析 Java 导入语句
        
        Args:
            code: Java 源代码
            
        Returns:
            Dict[str, str]: 导入映射表 {simple_name -> full_class_path}
        """
        import_map = {}
        
        try:
            from tree_sitter_languages import get_parser
            parser = get_parser('java')
            tree = parser.parse(bytes(code, 'utf8'))
            
            self._extract_java_import_nodes(tree.root_node, code, import_map)
        except Exception as e:
            # 如果 tree-sitter 解析失败，使用正则表达式回退
            import re
            # 匹配 import 语句
            pattern = r'import\s+(static\s+)?([\w.*]+)(?:\s+as\s+(\w+))?\s*;'
            for match in re.finditer(pattern, code):
                is_static = bool(match.group(1))
                full_path = match.group(2)
                alias = match.group(3)
                
                if '*' in full_path:
                    # 通配符导入，跳过
                    continue
                
                # 提取类名作为别名
                class_name = full_path.split('.')[-1]
                if alias:
                    import_map[alias] = full_path
                else:
                    import_map[class_name] = full_path
        
        return import_map
    
    def _extract_java_import_nodes(self, node, code: str, import_map: Dict[str, str]):
        """递归提取 Java 导入节点"""
        if node.type == 'import_declaration':
            full_path = None
            alias = None
            
            for child in node.children:
                if child.type == 'scoped_identifier' or child.type == 'identifier':
                    full_path = code[child.start_byte:child.end_byte]
                elif child.type == 'asterisk':
                    # 通配符导入，跳过
                    return
            
            if full_path and '*' not in full_path:
                class_name = full_path.split('.')[-1]
                import_map[class_name] = full_path
        
        for child in node.children:
            self._extract_java_import_nodes(child, code, import_map)
    
    def _parse_js_imports(self, code: str) -> Dict[str, str]:
        """
        解析 JavaScript/TypeScript 导入语句
        
        Args:
            code: JS/TS 源代码
            
        Returns:
            Dict[str, str]: 导入映射表 {local_name -> module_path}
        """
        import_map = {}
        
        try:
            from tree_sitter_languages import get_parser
            parser = get_parser('javascript')
            tree = parser.parse(bytes(code, 'utf8'))
            
            self._extract_js_import_nodes(tree.root_node, code, import_map)
        except Exception as e:
            print(f"TreeSitter JS 解析失败: {e}")
        
        # 无论 TreeSitter 是否成功，都使用正则表达式补充 CommonJS require
        import re
        
        # CommonJS require: const X = require('module')
        pattern3 = r'(?:const|let|var)\s+(\w+)\s*=\s*require\([\'"]([^\'"]+)[\'"]\)'
        for match in re.finditer(pattern3, code):
            name = match.group(1)
            module = match.group(2)
            import_map[name] = module
        
        return import_map
    
    def _extract_js_import_nodes(self, node, code: str, import_map: Dict[str, str]):
        """递归提取 JavaScript 导入节点"""
        if node.type == 'import_statement':
            self._process_js_import_node(node, code, import_map)
        
        for child in node.children:
            self._extract_js_import_nodes(child, code, import_map)
    
    def _process_js_import_node(self, node, code: str, import_map: Dict[str, str]):
        """处理单个 JavaScript 导入节点"""
        module_path = None
        imports = []
        
        for child in node.children:
            if child.type == 'string':
                # 提取模块路径（去掉引号）
                module_path = code[child.start_byte:child.end_byte].strip("'\"")
            elif child.type == 'import_clause':
                # 提取导入的名称
                for sub_child in child.children:
                    if sub_child.type == 'identifier':
                        imports.append((code[sub_child.start_byte:sub_child.end_byte], None))
                    elif sub_child.type == 'named_imports':
                        for spec in sub_child.children:
                            if spec.type == 'import_specifier':
                                name = None
                                alias = None
                                for spec_child in spec.children:
                                    if spec_child.type == 'identifier':
                                        if not name:
                                            name = code[spec_child.start_byte:spec_child.end_byte]
                                        else:
                                            alias = code[spec_child.start_byte:spec_child.end_byte]
                                if name:
                                    imports.append((name, alias))
        
        if module_path:
            for name, alias in imports:
                key = alias if alias else name
                import_map[key] = f"{module_path}/{name}"
    
    def _parse_c_imports(self, code: str) -> Dict[str, str]:
        """
        解析 C/C++ 包含指令
        
        Args:
            code: C/C++ 源代码
            
        Returns:
            Dict[str, str]: 导入映射表 {header_name -> header_path}
        """
        import_map = {}
        
        try:
            from tree_sitter_languages import get_parser
            parser = get_parser('c')
            tree = parser.parse(bytes(code, 'utf8'))
            
            self._extract_c_include_nodes(tree.root_node, code, import_map)
        except Exception as e:
            # 如果 tree-sitter 解析失败，使用正则表达式回退
            import re
            
            # 匹配 #include 指令
            pattern = r'#include\s*[<"]([^>"]+)[>"]'
            for match in re.finditer(pattern, code):
                header = match.group(1)
                # 提取文件名作为键
                header_name = header.split('/')[-1]
                import_map[header_name] = header
        
        return import_map
    
    def _extract_c_include_nodes(self, node, code: str, import_map: Dict[str, str]):
        """递归提取 C/C++ 包含节点"""
        if node.type == 'preproc_include':
            header_path = None
            
            for child in node.children:
                if child.type == 'system_lib_string' or child.type == 'string_literal':
                    # 提取头文件路径（去掉尖括号或引号）
                    header_path = code[child.start_byte:child.end_byte].strip('<>"')
            
            if header_path:
                # 提取文件名作为键
                header_name = header_path.split('/')[-1]
                import_map[header_name] = header_path
        
        for child in node.children:
            self._extract_c_include_nodes(child, code, import_map)
    
    def _extract_classes(self, code: str, language: str) -> List[Dict]:
        """
        提取文件中的类定义信息
        
        Args:
            code: 源代码
            language: 编程语言
            
        Returns:
            List[Dict]: 类信息列表，每个包含 name, start_line, end_line
        """
        classes = []
        
        try:
            from tree_sitter_languages import get_parser
            parser = get_parser(language)
            tree = parser.parse(bytes(code, 'utf8'))
            
            if language == 'python':
                self._extract_python_classes(tree.root_node, code, classes)
            elif language == 'java':
                self._extract_java_classes(tree.root_node, code, classes)
            elif language == 'javascript':
                self._extract_js_classes(tree.root_node, code, classes)
            # C/C++ 通常不使用类，或者使用 struct，这里简化处理
        except Exception as e:
            # 如果解析失败，返回空列表
            pass
        
        return classes
    
    def _extract_python_classes(self, node, code: str, classes: List[Dict]):
        """递归提取 Python 类定义"""
        if node.type == 'class_definition':
            class_name = None
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            for child in node.children:
                if child.type == 'identifier':
                    class_name = code[child.start_byte:child.end_byte]
                    break
            
            if class_name:
                classes.append({
                    'name': class_name,
                    'start_line': start_line,
                    'end_line': end_line
                })
        
        for child in node.children:
            self._extract_python_classes(child, code, classes)
    
    def _extract_java_classes(self, node, code: str, classes: List[Dict]):
        """递归提取 Java 类定义"""
        if node.type == 'class_declaration':
            class_name = None
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            for child in node.children:
                if child.type == 'identifier':
                    class_name = code[child.start_byte:child.end_byte]
                    break
            
            if class_name:
                classes.append({
                    'name': class_name,
                    'start_line': start_line,
                    'end_line': end_line
                })
        
        for child in node.children:
            self._extract_java_classes(child, code, classes)
    
    def _extract_js_classes(self, node, code: str, classes: List[Dict]):
        """递归提取 JavaScript 类定义"""
        if node.type == 'class_declaration':
            class_name = None
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            for child in node.children:
                if child.type == 'identifier':
                    class_name = code[child.start_byte:child.end_byte]
                    break
            
            if class_name:
                classes.append({
                    'name': class_name,
                    'start_line': start_line,
                    'end_line': end_line
                })
        
        for child in node.children:
            self._extract_js_classes(child, code, classes)
    
    def _find_enclosing_class(self, line_number: int, classes: List[Dict]) -> Optional[str]:
        """
        查找包含指定行号的类
        
        Args:
            line_number: 行号
            classes: 类信息列表
            
        Returns:
            Optional[str]: 类名，如果不在任何类中则返回 None
        """
        for cls in classes:
            if cls['start_line'] <= line_number <= cls['end_line']:
                return cls['name']
        return None
    
    def _extract_function_body(self, code: str, start_line: int, end_line: int) -> str:
        """
        从源代码中提取函数体（包含函数签名行）
        
        Args:
            code: 完整源代码
            start_line: 起始行号（1-based）
            end_line: 结束行号（1-based）
            
        Returns:
            str: 函数体代码
        """
        lines = code.split('\n')
        if start_line <= 0 or end_line > len(lines):
            return ""

        start_idx = start_line - 1
        end_idx = end_line
        
        body_lines = lines[start_idx:end_idx]
        return '\n'.join(body_lines)
    
    def _find_function_at_line(self, line_number: int, functions: List[Dict], 
                               file_path: str, classes: List[Dict]) -> Optional[str]:
        """
        查找包含指定行号的函数节点ID
        
        Args:
            line_number: 行号
            functions: 函数定义列表
            file_path: 文件路径
            classes: 类信息列表
            
        Returns:
            Optional[str]: 函数节点ID
        """
        for func in functions:
            if func['start_line'] <= line_number <= func['end_line']:
                func_name = func['name']
                class_name = self._find_enclosing_class(func['start_line'], classes)
                
                # 统一节点ID格式
                return f"{file_path}::{class_name or ''}::{func_name}"
        return None
    
    def _resolve_callee_node_id(self, called_func_name: str, caller_file_path: str,
                                caller_classes: List[Dict], call_line: int,
                                import_map: Dict[str, str] = None) -> Optional[str]:
        """
        解析被调用函数的节点ID
        
        Args:
            called_func_name: 被调用的函数名
            caller_file_path: 调用者文件路径
            caller_classes: 调用者所在文件的类信息
            call_line: 调用发生的行号
            import_map: 导入映射表
            
        Returns:
            Optional[str]: 被调用函数的节点ID
        """
        import_map = import_map or {}
        
        # 情况1：如果是方法调用（obj.method），尝试在同文件中查找
        if '.' in called_func_name:
            parts = called_func_name.split('.')
            obj_name = parts[0]
            method_name = parts[-1]
            
            # 先尝试在当前类的方法中查找
            current_class = self._find_enclosing_class(call_line, caller_classes)
            if current_class:
                candidate_id = f"{caller_file_path}::{current_class}::{method_name}"
                if candidate_id in self.function_metadata:
                    return candidate_id
            
            # 检查是否是导入的模块
            if obj_name in import_map:
                # 这是一个导入的模块，尝试在对应的文件中查找
                module_path = import_map[obj_name]
                # 将模块路径转换为文件路径
                potential_file = module_path.replace('.', '/') + '.py'
                
                # 在所有已知函数中查找
                for node_id in self.function_metadata:
                    if (potential_file in node_id and 
                        node_id.endswith(f"::{method_name}")):
                        return node_id
            
            # 再尝试在同文件的其他位置查找
            for node_id in self.function_metadata:
                if node_id.startswith(caller_file_path + "::") and node_id.endswith(f"::{method_name}"):
                    return node_id
        else:
            # 情况2：普通函数调用
            # 首先尝试在同文件中查找
            for node_id in self.function_metadata:
                if node_id.startswith(caller_file_path + "::") and node_id.endswith(f"::{called_func_name}"):
                    return node_id
            
            # 检查是否是导入的函数
            if called_func_name in import_map:
                # 这是一个导入的函数，尝试在对应的文件中查找
                module_path = import_map[called_func_name]
                potential_file = module_path.replace('.', '/') + '.py'
                
                # 在所有已知函数中查找
                for node_id in self.function_metadata:
                    if (potential_file in node_id and 
                        node_id.endswith(f"::{called_func_name}")):
                        return node_id
            
            # 如果在同文件中没找到，尝试在所有已知的函数中查找（可能是跨文件调用）
            candidates = [nid for nid in self.function_metadata 
                         if self.function_metadata[nid].name == called_func_name]
            
            if len(candidates) == 1:
                # 只有一个候选，直接返回
                return candidates[0]
            elif len(candidates) > 1:
                # 多个候选，优先选择同模块的
                caller_module = Path(caller_file_path).stem
                for candidate in candidates:
                    candidate_module = Path(self.function_metadata[candidate].file_path).stem
                    if candidate_module == caller_module:
                        return candidate
                
                # 如果还是没有明确的，返回第一个（可能有歧义）
                return candidates[0]
        
        return None
    
    def save_call_graph(self, output_path: str):
        """
        保存调用图到文件
        
        Args:
            output_path: 输出文件路径（.pkl）
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.call_graph, f)
        
        print(f"调用图已保存到: {output_path}")
    
    def load_call_graph(self, input_path: str) -> nx.DiGraph:
        """
        从文件加载调用图
        
        Args:
            input_path: 输入文件路径（.pkl）
            
        Returns:
            nx.DiGraph: 调用图
        """
        with open(input_path, 'rb') as f:
            self.call_graph = pickle.load(f)
        
        print(f"调用图已从 {input_path} 加载")
        return self.call_graph
    
    def save_function_metadata(self, output_path: str):
        """
        保存函数元数据到 JSON 文件
        
        Args:
            output_path: 输出文件路径（.json）
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为可序列化的字典
        metadata_dict = {
            node_id: metadata.to_dict() 
            for node_id, metadata in self.function_metadata.items()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        
        print(f"函数元数据已保存到: {output_path}")
    
    def load_function_metadata(self, input_path: str) -> Dict[str, FunctionMetadata]:
        """
        从 JSON 文件加载函数元数据
        
        Args:
            input_path: 输入文件路径（.json）
            
        Returns:
            Dict[str, FunctionMetadata]: 函数元数据字典
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)
        
        self.function_metadata = {
            node_id: FunctionMetadata.from_dict(data)
            for node_id, data in metadata_dict.items()
        }
        
        print(f"函数元数据已从 {input_path} 加载")
        return self.function_metadata
    
    def get_statistics(self) -> Dict:
        """
        获取静态分析的统计信息
        
        Returns:
            Dict: 统计信息
        """
        stats = {
            'total_functions': len(self.function_metadata),
            'total_nodes': self.call_graph.number_of_nodes(),
            'total_edges': self.call_graph.number_of_edges(),
            'languages': {},
            'files_with_functions': set(),
            'classes_with_methods': set(),
            'avg_args_per_function': 0,
            'avg_function_size': 0
        }
        
        total_args = 0
        total_size = 0
        
        for node_id, metadata in self.function_metadata.items():
            # 语言统计
            lang = metadata.language
            stats['languages'][lang] = stats['languages'].get(lang, 0) + 1
            
            # 文件统计
            stats['files_with_functions'].add(metadata.file_path)
            
            # 类统计
            if metadata.class_name:
                stats['classes_with_methods'].add(f"{metadata.file_path}::{metadata.class_name}")
            
            # 参数和大小统计
            total_args += metadata.arg_count
            total_size += (metadata.end_line - metadata.start_line)
        
        # 计算平均值
        if stats['total_functions'] > 0:
            stats['avg_args_per_function'] = round(total_args / stats['total_functions'], 2)
            stats['avg_function_size'] = round(total_size / stats['total_functions'], 2)
        
        # 转换集合为列表以便序列化
        stats['files_with_functions'] = len(stats['files_with_functions'])
        stats['classes_with_methods'] = len(stats['classes_with_methods'])
        
        return stats


def run_static_analysis(source_code_dir: str, project_name: str, 
                       projects_dir: str = "./projects"):
    """
    运行静态代码分析的主函数
    
    Args:
        source_code_dir: 源代码目录路径
        project_name: 项目名称
        projects_dir: 项目根目录
        
    Returns:
        Tuple[nx.DiGraph, Dict[str, FunctionMetadata]]: 调用图和函数元数据
    """
    # 构建输出目录：projects/<project_name>/static_analysis/
    output_dir = Path(projects_dir) / project_name / "static_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建分析器
    analyzer = StaticCodeAnalyzer(source_code_dir, projects_dir)
    
    # 执行分析
    call_graph, function_metadata = analyzer.analyze_all_files()
    
    # 保存结果到项目专属目录
    call_graph_path = output_dir / "call_graph.pkl"
    metadata_path = output_dir / "function_metadata.json"
    
    analyzer.save_call_graph(str(call_graph_path))
    analyzer.save_function_metadata(str(metadata_path))
    
    # 打印统计信息
    stats = analyzer.get_statistics()
    print("\n" + "="*60)
    print("静态代码分析统计:")
    print(f"- 总函数数: {stats['total_functions']}")
    print(f"- 调用图节点数: {stats['total_nodes']}")
    print(f"- 调用图边数: {stats['total_edges']}")
    print(f"- 语言分布: {stats['languages']}")
    print(f"- 包含函数的文件数: {stats['files_with_functions']}")
    print(f"- 包含方法的类数: {stats['classes_with_methods']}")
    print(f"- 平均每函数参数数: {stats['avg_args_per_function']}")
    print(f"- 平均函数大小(行): {stats['avg_function_size']}")
    print(f"- 输出目录: {output_dir}")
    print("="*60)
    
    return call_graph, function_metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="静态代码分析")
    parser.add_argument("--source-dir", type=str, required=True, 
                       help="源代码目录路径")
    parser.add_argument("--project-name", type=str, required=True,
                       help="项目名称")
    parser.add_argument("--projects-dir", type=str, default="./projects",
                       help="项目根目录")
    
    args = parser.parse_args()
    
    call_graph, function_metadata = run_static_analysis(
        source_code_dir=args.source_dir,
        project_name=args.project_name,
        projects_dir=args.projects_dir
    )
    
    print("静态代码分析完成")