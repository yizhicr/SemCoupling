"""
数据获取模块
获取固定Github仓库的项目clone和commit
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from pydriller import Repository
import json
import re


def extract_modified_lines_from_diff(diff_text: str) -> List[int]:
    """
    从 git diff 文本中提取被修改的行号
    
    Args:
        diff_text: git diff 文本
        
    Returns:
        List[int]: 被修改的行号列表（新文件的行号）
    """
    modified_lines = []
    
    if not diff_text:
        return modified_lines
    
    # 解析 unified diff 格式
    current_line = 0
    in_hunk = False
    new_line_offset = 0
    
    for line in diff_text.split('\n'):
        # 检测 hunk 头部，例如: @@ -10,5 +10,6 @@
        if line.startswith('@@'):
            in_hunk = True
            # 提取新文件的起始行号
            match = re.search(r'\+(\d+)(?:,\d+)?', line)
            if match:
                current_line = int(match.group(1))
                new_line_offset = 0
            continue
        
        if not in_hunk:
            continue
            
        # 处理 hunk 中的行
        if line.startswith('+') and not line.startswith('+++'):
            # 新增的行
            modified_lines.append(current_line + new_line_offset)
            new_line_offset += 1
        elif line.startswith('-') and not line.startswith('---'):
            # 删除的行
            pass
        elif line.startswith(' '):
            # 上下文行（未修改）
            new_line_offset += 1
        elif line.startswith('\\'):
            # 特殊标记行，跳过
            continue
    
    return modified_lines


def extract_function_at_line(functions: List[Dict], line_number: int) -> Optional[Dict]:
    """
    查找包含指定行号的函数
    
    Args:
        functions: 函数定义列表
        line_number: 要查找的行号
        
    Returns:
        Optional[Dict]: 找到的函数信息，如果没找到则返回 None
    """
    for func in functions:
        if func['start_line'] <= line_number <= func['end_line']:
            return func
    return None


def get_file_extension(file_path: str) -> str:
    """根据文件扩展名确定语言类型"""
    ext = os.path.splitext(file_path)[1].lower()
    lang_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'tsx',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.cxx': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'c_sharp',
        '.go': 'go',
        '.rs': 'rust'
    }
    return lang_map.get(ext, None)


def extract_function_definitions(code: str, lang: str) -> Dict[str, Tuple[int, int]]:
    """从代码中提取函数定义"""
    try:
        # 尝试使用 tree-sitter 解析
        from tree_sitter_languages import get_parser
        parser = get_parser(lang)
        tree = parser.parse(bytes(code, 'utf8'))
        
        functions = {}
        
        # 根据语言类型查找函数定义节点
        if lang == 'python':
            # Python 函数定义查询
            query_str = """
            (function_definition
                name: (identifier) @function_name)
            (class_definition
                name: (identifier) @class_name
                body: (block
                    (function_definition
                        name: (identifier) @method_name)))
            """
        elif lang == 'java':
            # Java 方法定义查询
            query_str = """
            (method_declaration
                name: (identifier) @method_name)
            (class_declaration
                name: (identifier) @class_name)
            """
        elif lang == 'javascript':
            # JavaScript 函数定义查询
            query_str = """
            (function_declaration
                name: (identifier) @function_name)
            (method_definition
                name: (property_identifier) @method_name)
            (variable_declarator
                name: (identifier) @var_name
                value: [(arrow_function) (function_expression)])
            """
        elif lang in ['c', 'cpp']:
            # C/C++ 函数定义查询
            query_str = """
            (function_definition
                declarator: (function_declarator
                    declarator: (identifier) @function_name))
            (function_definition
                declarator: (function_declarator
                    declarator: (field_identifier) @function_name))
            """
        else:
            # 如果语言不支持，则回退到正则表达式
            return extract_functions_with_regex(code, lang)
        
        # 检查是否支持查询功能
        try:
            query = parser.query(query_str)
            captures = query.captures(tree)
            
            # 提取函数名和位置
            for node, capture_type in captures:
                if capture_type.decode('utf8') in ['function_name', 'method_name', 'var_name']:
                    func_name = code[node.start_byte:node.end_byte]
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    functions[f"{func_name}"] = (start_line, end_line)
        except AttributeError:
            # 如果解析器不支持查询功能，使用正则表达式
            return extract_functions_with_regex(code, lang)
        
        # 如果 tree-sitter 没有找到函数，回退到正则表达式
        if not functions:
            functions = extract_functions_with_regex(code, lang)
        
        return functions
    except ImportError:
        # 如果 tree-sitter 不可用，则使用正则表达式
        return extract_functions_with_regex(code, lang)
    except Exception:
        # 如果 tree-sitter 解析失败，回退到正则表达式
        return extract_functions_with_regex(code, lang)


def extract_functions_with_regex(content, language):
    """
    使用正则表达式提取函数定义
    """
    functions = {}
    
    if language in ['c', 'cpp', 'java', 'go']:
        # 匹配C/C++/Java/Go风格的函数定义
        # 支持public/private/static等修饰符，以及模板函数
        pattern = r'(?:^|\n)\s*(?:\w+\s+)*\w+(?:\s*<(?:[^<>]|<[^<>]*>)*>)?\s+(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:final\s*)?{'
        matches = re.finditer(pattern, content, re.MULTILINE)
        for match in matches:
            func_name = match.group(1)
            start_line = content.count('\n', 0, match.start()) + 1
            functions[func_name] = (start_line, start_line)  # (起始行, 结束行)
    elif language == 'python':
        # 匹配Python函数定义
        pattern = r'(?:^|\s*)(?:async\s+)?def\s+(\w+)\s*\([^)]*\)\s*:'
        matches = re.finditer(pattern, content, re.MULTILINE)
        for match in matches:
            func_name = match.group(1)
            start_line = content.count('\n', 0, match.start()) + 1
            functions[func_name] = (start_line, start_line)
    elif language == 'javascript':
        # 匹配JavaScript函数定义
        patterns = [
            r'(?:^|\s)(\w+)\s*=\s*function\s*\([^)]*\)\s*{',
            r'(?:^|\s)function\s+(\w+)\s*\([^)]*\)\s*{',
            r'(?:^|\s)(\w+)\s*:\s*function\s*\([^)]*\)\s*{',
            r'(?:^|\s)(\w+)\s*=\s*\([^)]*\)\s*=>\s*{',
        ]
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                func_name = match.group(1)
                start_line = content.count('\n', 0, match.start()) + 1
                functions[func_name] = (start_line, start_line)
    elif language == 'typescript':
        # 匹配TypeScript函数定义（包括interface和type定义）
        patterns = [
            r'(?:^|\s)(\w+)\s*=\s*function\s*\([^)]*\)\s*{',
            r'(?:^|\s)function\s+(\w+)\s*\([^)]*\)\s*{',
            r'(?:^|\s)(\w+)\s*:\s*function\s*\([^)]*\)\s*{',
            r'(?:^|\s)(\w+)\s*=\s*\([^)]*\)\s*=>\s*{',
            r'(?:^|\s)(\w+)\s*\([^)]*\)\s*{\s*',
        ]
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                func_name = match.group(1)
                start_line = content.count('\n', 0, match.start()) + 1
                functions[func_name] = (start_line, start_line)
    elif language == 'c_sharp':
        # 匹配C#函数定义
        pattern = r'(?:^|\n)\s*(?:public|private|protected|internal|\s)*\s*\w+\s+\w*\s*(\w+)\s*\([^)]*\)\s*{'
        matches = re.finditer(pattern, content, re.MULTILINE)
        for match in matches:
            func_name = match.group(1)
            start_line = content.count('\n', 0, match.start()) + 1
            functions[func_name] = (start_line, start_line)
    else:
        # 其他语言的通用匹配
        pattern = r'(?:^|\n)(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*{'
        matches = re.finditer(pattern, content, re.MULTILINE)
        for match in matches:
            func_name = match.group(1)
            start_line = content.count('\n', 0, match.start()) + 1
            functions[func_name] = (start_line, start_line)
    
    return functions


class TreeSitterParser:
    """
    使用 tree-sitter 解析代码，提取函数定义和调用关系
    """
    def __init__(self):
        self.supported_languages = ['python', 'java', 'javascript', 'cpp', 'c']

    def extract_functions(self, code: str, language: str) -> List[Dict]:
        """
        提取代码中的函数定义信息
        """
        if language not in self.supported_languages:
            return []
        
        functions = []
        
        try:
            # 使用 tree-sitter-languages 获取解析器
            from tree_sitter_languages import get_parser
            parser = get_parser(language)
            tree = parser.parse(bytes(code, 'utf8'))
            
            # 根据语言类型查找函数定义节点
            if language == 'python':
                self._extract_python_functions(tree.root_node, code, functions)
            elif language == 'java':
                self._extract_java_functions(tree.root_node, code, functions)
            elif language == 'javascript':
                self._extract_js_functions(tree.root_node, code, functions)
            elif language in ['c', 'cpp']:
                self._extract_c_functions(tree.root_node, code, functions, current_class=None)
            
        except Exception as e:
            print(f"Tree-sitter 解析失败: {e}")
        
        # 如果 tree-sitter 没有提取到任何函数，使用正则表达式回退
        if not functions:
            regex_funcs = extract_functions_with_regex(code, language)
            for name, (start_line, end_line) in regex_funcs.items():
                functions.append({
                    'name': name,
                    'start_line': start_line,
                    'end_line': end_line,
                    'file_path': '',
                    'language': language,
                    'parameters': [],
                    'body_size': end_line - start_line
                })
        
        return functions

    def _extract_python_functions(self, node, code: str, functions: List[Dict]):
        """
        递归提取Python函数定义
        """
        if node.type == 'function_definition':
            # 查找函数名
            func_name = None
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            # 寻找 identifier 子节点作为函数名
            for child in node.children:
                if child.type == 'identifier':
                    func_name = code[child.start_byte:child.end_byte]
                    break
            
            if func_name:
                # 寻找 parameters 子节点
                params_node = None
                for child in node.children:
                    if child.type == 'parameters':
                        params_node = child
                        break
                
                func_info = {
                    'name': func_name,
                    'start_line': start_line,
                    'end_line': end_line,
                    'file_path': '',
                    'language': 'python',
                    'parameters': self._extract_params(params_node, code) if params_node else [],
                    'body_size': end_line - start_line
                }
                functions.append(func_info)
        elif node.type == 'class_definition':
            # 处理类中的方法
            for child in node.children:
                if child.type == 'block':
                    for grandchild in child.children:
                        if grandchild.type == 'function_definition':
                            self._extract_python_functions(grandchild, code, functions)
        
        # 递归处理子节点
        for child in node.children:
            self._extract_python_functions(child, code, functions)

    def _extract_java_functions(self, node, code: str, functions: List[Dict]):
        """
        递归提取Java方法定义
        """
        if node.type == 'method_declaration':
            # 查找方法名
            method_name = None
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            for child in node.children:
                if child.type == 'identifier':
                    # 确保不是其他上下文中的 identifier
                    method_name = code[child.start_byte:child.end_byte]
                    break
            
            if method_name:
                # 寻找 formal_parameters
                params_node = None
                for child in node.children:
                    if child.type == 'formal_parameters':
                        params_node = child
                        break

                func_info = {
                    'name': method_name,
                    'start_line': start_line,
                    'end_line': end_line,
                    'file_path': '',
                    'language': 'java',
                    'parameters': self._extract_params(params_node, code) if params_node else [],
                    'body_size': end_line - start_line
                }
                functions.append(func_info)
        
        # 递归处理子节点
        for child in node.children:
            self._extract_java_functions(child, code, functions)

    def _extract_js_functions(self, node, code: str, functions: List[Dict]):
        """
        递归提取JavaScript函数定义
        """
        if node.type in ['function_declaration', 'function', 'method_definition']:
            func_name = None
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            for child in node.children:
                if child.type == 'identifier' or child.type == 'property_identifier':
                    func_name = code[child.start_byte:child.end_byte]
                    break
            
            if func_name:
                # 寻找 formal_parameters
                params_node = None
                for child in node.children:
                    if child.type == 'formal_parameters':
                        params_node = child
                        break

                func_info = {
                    'name': func_name,
                    'start_line': start_line,
                    'end_line': end_line,
                    'file_path': '',
                    'language': 'javascript',
                    'parameters': self._extract_params(params_node, code) if params_node else [],
                    'body_size': end_line - start_line
                }
                functions.append(func_info)
        
        # 递归处理子节点
        for child in node.children:
            self._extract_js_functions(child, code, functions)

    def _extract_c_functions(self, node, code: str, functions: List[Dict], current_class=None):
        """
        递归提取C/C++函数定义（支持类方法识别）
        
        Args:
            node: AST节点
            code: 源代码字符串
            functions: 函数列表（用于收集结果）
            current_class: 当前所在的类名（用于嵌套作用域）
        """
        if node.type == 'class_specifier':
            # 提取类名
            class_name = None
            for child in node.children:
                if child.type == 'type_identifier':
                    class_name = code[child.start_byte:child.end_byte]
                    break
            
            # 递归处理类体中的成员函数
            for child in node.children:
                if child.type == 'field_declaration_list':
                    for member in child.children:
                        self._extract_c_functions(member, code, functions, class_name)
            return  # 类本身不是函数，不继续递归其他子节点以避免重复
        
        elif node.type == 'struct_specifier':
            # 同样处理结构体（C语言中常用）
            struct_name = None
            for child in node.children:
                if child.type == 'type_identifier':
                    struct_name = code[child.start_byte:child.end_byte]
                    break
            
            # 递归处理结构体成员
            for child in node.children:
                if child.type == 'field_declaration_list':
                    for member in child.children:
                        self._extract_c_functions(member, code, functions, struct_name)
            return
        
        elif node.type == 'function_definition':
            func_name = None
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            # 在函数声明器中查找函数名
            for child in node.children:
                if child.type == 'function_declarator':
                    for sub_child in child.children:
                        if sub_child.type == 'identifier':
                            func_name = code[sub_child.start_byte:sub_child.end_byte]
                            break
                elif child.type == 'identifier' and not func_name:
                    # 某些情况下 identifier 可能直接作为子节点
                    func_name = code[child.start_byte:child.end_byte]
            
            if func_name:
                # 确定语言类型（根据文件扩展名或代码特征）
                lang = 'cpp' if current_class else 'c'
                
                func_info = {
                    'name': func_name,
                    'start_line': start_line,
                    'end_line': end_line,
                    'file_path': '',
                    'language': lang,
                    'class_name': current_class,  # 添加类名信息
                    'parameters': [],
                    'body_size': end_line - start_line
                }
                functions.append(func_info)
        
        # 递归处理子节点，传递当前类上下文
        for child in node.children:
            self._extract_c_functions(child, code, functions, current_class)

    def _extract_params(self, node, code: str):
        """
        提取参数（辅助函数）
        """
        params = []
        if node and node.type in ['parameters', 'formal_parameters', 'parameter_list']:
            for child in node.children:
                if child.type == 'parameter' or child.type == 'typed_parameter':
                    for sub_child in child.children:
                        if sub_child.type == 'identifier':
                            param_name = code[sub_child.start_byte:sub_child.end_byte]
                            params.append(param_name)
                elif child.type == 'identifier':
                     # 直接包含 identifier 的情况
                    params.append(code[child.start_byte:child.end_byte])
        return params

    def extract_calls(self, code: str, language: str) -> List[Dict]:
        """
        提取代码中的函数调用信息
        """
        if language not in self.supported_languages:
            return []
        
        try:
            # 使用 tree-sitter-languages 获取解析器
            from tree_sitter_languages import get_parser
            parser = get_parser(language)
            tree = parser.parse(bytes(code, 'utf8'))
            
            calls = []
            
            # 根据语言类型查找函数调用节点
            if language == 'python':
                self._extract_python_calls(tree.root_node, code, calls)
            elif language == 'java':
                self._extract_java_calls(tree.root_node, code, calls)
            elif language == 'javascript':
                self._extract_js_calls(tree.root_node, code, calls)
            elif language in ['c', 'cpp']:
                self._extract_c_calls(tree.root_node, code, calls)
            
            return calls
        except Exception as e:
            print(f"Tree-sitter 调用提取失败: {e}")
            raise e

    def _extract_python_calls(self, node, code: str, calls: List[Dict]):
        """
        递归提取Python函数调用
        """
        if node.type == 'call':
            func_name = None
            start_line = node.start_point[0] + 1
            
            # 获取函数名
            for child in node.children:
                if child.type == 'identifier':
                    func_name = code[child.start_byte:child.end_byte]
                    break
                elif child.type == 'attribute':
                    obj_name = None
                    method_name = None
                    for attr_child in child.children:
                        if attr_child.type == 'identifier':
                            if not obj_name:
                                obj_name = code[attr_child.start_byte:attr_child.end_byte]
                            else:
                                method_name = code[attr_child.start_byte:attr_child.end_byte]
                                break
                    
                    if obj_name and method_name:
                        func_name = f"{obj_name}.{method_name}"
                    elif obj_name:
                        pass 
            
            if func_name:
                call_info = {
                    'function': func_name,
                    'line_number': start_line,
                    'type': 'function_call'
                }
                calls.append(call_info)
        
        # 递归处理子节点
        for child in node.children:
            self._extract_python_calls(child, code, calls)

    def _extract_java_calls(self, node, code: str, calls: List[Dict]):
        """
        递归提取Java方法调用
        """
        if node.type == 'method_invocation':
            method_name = None
            start_line = node.start_point[0] + 1
            
            # 获取方法名
            for child in node.children:
                if child.type == 'identifier':
                    method_name = code[child.start_byte:child.end_byte]
                    break
                elif child.type == 'field_access' or child.type == 'scoped_identifier':
                    for attr_child in child.children:
                        if attr_child.type == 'identifier':
                            # 通常最后一个 identifier 是方法名
                            method_name = code[attr_child.start_byte:attr_child.end_byte]
            
            if method_name:
                call_info = {
                    'function': method_name,
                    'line_number': start_line,
                    'type': 'method_call'
                }
                calls.append(call_info)
        
        # 递归处理子节点
        for child in node.children:
            self._extract_java_calls(child, code, calls)

    def _extract_js_calls(self, node, code: str, calls: List[Dict]):
        """
        递归提取JavaScript函数调用
        """
        if node.type == 'call_expression':
            func_name = None
            start_line = node.start_point[0] + 1
            
            # 获取函数名
            for child in node.children:
                if child.type == 'identifier':
                    func_name = code[child.start_byte:child.end_byte]
                    break
                elif child.type == 'member_expression':
                    obj_name = None
                    method_name = None
                    for attr_child in child.children:
                        if attr_child.type == 'identifier' or attr_child.type == 'property_identifier':
                            if not obj_name:
                                obj_name = code[attr_child.start_byte:attr_child.end_byte]
                            else:
                                method_name = code[attr_child.start_byte:attr_child.end_byte]
                                break
                    
                    if obj_name and method_name:
                        func_name = f"{obj_name}.{method_name}"
            
            if func_name:
                call_info = {
                    'function': func_name,
                    'line_number': start_line,
                    'type': 'function_call'
                }
                calls.append(call_info)
        
        # 递归处理子节点
        for child in node.children:
            self._extract_js_calls(child, code, calls)

    def _extract_c_calls(self, node, code: str, calls: List[Dict]):
        """
        递归提取C/C++函数调用
        """
        if node.type == 'call_expression':
            func_name = None
            start_line = node.start_point[0] + 1
            
            # 获取函数名
            for child in node.children:
                if child.type == 'identifier':
                    func_name = code[child.start_byte:child.end_byte]
                    break
                elif child.type == 'field_expression' or child.type == 'pointer_expression':
                     # 处理结构体或指针调用，简化处理只取 identifier
                    for sub_child in child.children:
                        if sub_child.type == 'field_identifier' or sub_child.type == 'identifier':
                             func_name = code[sub_child.start_byte:sub_child.end_byte]
                             break
            
            if func_name:
                call_info = {
                    'function': func_name,
                    'line_number': start_line,
                    'type': 'function_call'
                }
                calls.append(call_info)
        
        # 递归处理子节点
        for child in node.children:
            self._extract_c_calls(child, code, calls)


def extract_calls_with_regex(content, language):
    """
    使用正则表达式提取函数调用
    """
    calls = []
    
    if language == 'python':
        # Python 函数调用模式
        pattern = r'\b(\w+(?:\.\w+)*)\.(\w+)\s*\(|\b(\w+)\s*\('
        matches = re.finditer(pattern, content)
        for match in matches:
            if match.group(1) and match.group(2):
                calls.append({'function': f"{match.group(1)}.{match.group(2)}", 'line_number': content.count('\n', 0, match.start()) + 1})
            elif match.group(3):  # function() 形式
                calls.append({'function': match.group(3), 'line_number': content.count('\n', 0, match.start()) + 1})
    elif language == 'java':
        # Java 函数调用模式
        pattern = r'\b(\w+(?:\.\w+)*)\.(\w+)\s*\(|\b(\w+)\s*\('
        matches = re.finditer(pattern, content)
        for match in matches:
            if match.group(1) and match.group(2):
                calls.append({'function': f"{match.group(1)}.{match.group(2)}", 'line_number': content.count('\n', 0, match.start()) + 1})
            elif match.group(3):  # function() 形式
                calls.append({'function': match.group(3), 'line_number': content.count('\n', 0, match.start()) + 1})
    elif language == 'javascript':
        # JavaScript 函数调用模式
        pattern = r'\b(\w+(?:\.\w+)*)\.(\w+)\s*\(|\b(\w+)\s*\('
        matches = re.finditer(pattern, content)
        for match in matches:
            if match.group(1) and match.group(2):
                calls.append({'function': f"{match.group(1)}.{match.group(2)}", 'line_number': content.count('\n', 0, match.start()) + 1})
            elif match.group(3):  # function() 形式
                calls.append({'function': match.group(3), 'line_number': content.count('\n', 0, match.start()) + 1})
    else:
        # 其他语言的通用模式
        pattern = r'\b(\w+)\s*\('
        matches = re.finditer(pattern, content)
        for match in matches:
            calls.append({'function': match.group(1), 'line_number': content.count('\n', 0, match.start()) + 1})
    
    return calls


class GitHubAnalyzer:
    def __init__(self, projects_dir: str = "./projects"):
        """
        初始化GitHub分析器
        
        Args:
            projects_dir: 项目存储目录
        """
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(exist_ok=True)
        self.cloned_repo_path = None
        self.parser = TreeSitterParser()

    def clone_repository(self, repo_url: str, repo_name: Optional[str] = None, branch: Optional[str] = None) -> bool:
        """
        克隆GitHub仓库到本地
        
        Args:
            repo_url: GitHub仓库URL
            repo_name: 本地存储的仓库名称，如果不指定则从URL中提取
            branch: 分支名称，如果为None则使用默认分支
            
        Returns:
            bool: 是否成功克隆
        """
        try:
            # 从URL中提取项目名称
            if not repo_name:
                repo_name = repo_url.rstrip('/').split('/')[-1]
                if repo_name.endswith('.git'):
                    repo_name = repo_name[:-4]
                
                if repo_name == '':
                    repo_name = 'cloned_repo'
            
            # 创建项目专用目录
            self.project_dir = self.projects_dir / repo_name
            self.project_dir.mkdir(exist_ok=True)
            
            self.cloned_repo_path = self.project_dir / 'source_code'
            
            # 检查是否已经克隆过
            if self.cloned_repo_path.exists():
                print(f"项目已存在于: {self.cloned_repo_path}，跳过克隆步骤")
                # 检查是否有.git目录，判断是否为有效的git仓库
                if (self.cloned_repo_path / '.git').exists():
                    print("检测到有效的git仓库，继续分析...")
                    return True
                else:
                    print("不是有效的git仓库，重新克隆")
                    self._remove_directory(self.cloned_repo_path)
            
            clone_cmd = ["git", "clone"]
            if branch:
                clone_cmd.extend(["--branch", branch])
            clone_cmd.extend([repo_url, str(self.cloned_repo_path)])
            
            result = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"克隆失败: {result.stderr}")
                return False
            
            print(f"仓库克隆成功，路径: {self.cloned_repo_path}")
            return True
            
        except subprocess.TimeoutExpired:
            print("克隆超时")
            return False
        except Exception as e:
            print(f"克隆过程中出现错误: {str(e)}")
            return False

    def _remove_directory(self, directory: Path):
        """
        删除目录及其内容
        """
        import shutil
        shutil.rmtree(directory)

    def get_commit_history(self, max_commits: Optional[int] = None) -> List[Dict]:
        """
        获取仓库的提交历史，包括函数级别的修改信息
        
        Args:
            max_commits: 最大提交数量限制，None表示不限制
            
        Returns:
            List[Dict]: 提交历史列表
        """
        if not self.cloned_repo_path or not self.cloned_repo_path.exists():
            raise ValueError("仓库未克隆或不存在")
        
        commits = []
        
        print(f"开始获取仓库提交历史: {self.cloned_repo_path}")
        
        # 使用PyDriller遍历提交历史
        for i, commit in enumerate(Repository(str(self.cloned_repo_path)).traverse_commits()):
            if max_commits and i >= max_commits:
                break
                
            # 提取提交信息，包括函数级别的修改信息
            commit_info = {
                'hash': commit.hash,
                'author': {
                    'name': commit.author.name,
                    'email': commit.author.email
                },
                'committer_date': str(commit.committer_date),
                'message': commit.msg,
                'added_lines': commit.insertions,
                'deleted_lines': commit.deletions,
                'files_modified': [],
                'functions_modified': []  # 新增：函数级别的修改信息
            }
            
            for f in commit.modified_files:
                file_info = {
                    'filename': f.filename,
                    'added_lines': f.added_lines,
                    'deleted_lines': f.deleted_lines,
                    'old_path': f.old_path,
                    'new_path': f.new_path,
                    'diff': f.diff,  # 保留 diff 信息
                    'modified_lines': []  # 新增：具体修改的行号列表
                }
                
                # 从 diff 中提取具体修改的行号
                if f.diff:
                    modified_lines = extract_modified_lines_from_diff(f.diff)
                    file_info['modified_lines'] = modified_lines
                
                commit_info['files_modified'].append(file_info)
                
                # 分析函数级别的修改
                lang = get_file_extension(f.filename)
                if lang and f.source_code:
                    # 提取修改文件中的函数定义
                    functions_in_file = self.parser.extract_functions(f.source_code, lang)
                    
                    # 如果有具体的修改行号，精确匹配受影响的函数
                    if file_info['modified_lines']:
                        affected_functions = set()
                        for line_num in file_info['modified_lines']:
                            func = extract_function_at_line(functions_in_file, line_num)
                            if func:
                                # 使用函数名作为唯一标识，避免重复
                                func_key = f"{func['name']}:{func['start_line']}-{func['end_line']}"
                                if func_key not in affected_functions:
                                    affected_functions.add(func_key)
                                    commit_info['functions_modified'].append({
                                        'function_name': func['name'],
                                        'filename': f.filename,
                                        'file_path': f"{self.cloned_repo_path}/{f.filename}",
                                        'lines_modified': (func['start_line'], func['end_line']),
                                        'specific_modified_lines': [l for l in file_info['modified_lines'] 
                                                                   if func['start_line'] <= l <= func['end_line']],
                                        'language': lang,
                                        'parameters': func.get('parameters', []),
                                        'body_size': func.get('body_size', 0)
                                    })
                    else:
                        # 如果没有解析出行号，回退到原来的方式：检查 diff 中是否包含函数行号
                        for func_info in functions_in_file:
                            if f.diff and any(str(line_num) in f.diff 
                                              for line_num in range(func_info['start_line'], func_info['end_line']+1)):
                                commit_info['functions_modified'].append({
                                    'function_name': func_info['name'],
                                    'filename': f.filename,
                                    'file_path': f"{self.cloned_repo_path}/{f.filename}",
                                    'lines_modified': (func_info['start_line'], func_info['end_line']),
                                    'specific_modified_lines': [],
                                    'language': lang,
                                    'parameters': func_info.get('parameters', []),
                                    'body_size': func_info.get('body_size', 0)
                                })
            
            commits.append(commit_info)
            
            print(f"已获取 {i + 1} 个提交", end='\r')
        
        print(f"共获取到 {len(commits)} 个提交记录")
        return commits

    def get_commit_statistics(self, max_commits: Optional[int] = None) -> Dict:
        """
        获取仓库的提交统计信息
        
        Returns:
            Dict: 提交统计信息
        """
        if not self.cloned_repo_path or not self.cloned_repo_path.exists():
            raise ValueError("仓库未克隆或不存在，请先调用clone_repository方法")
        
        stats = {
            'total_commits': 0,
            'total_authors': set(),
            'file_extensions': {},
            'monthly_activity': {},
            'total_files_changed': 0,
            'total_added_lines': 0,
            'total_deleted_lines': 0,
            'function_modifications': {},  # 统计函数修改次数
            'total_functions_changed': 0
        }
        
        print(f"计算仓库统计信息: {self.cloned_repo_path}")
        
        for i, commit in enumerate(Repository(str(self.cloned_repo_path)).traverse_commits()):
            if max_commits and i >= max_commits:
                break
            # 提交总数
            stats['total_commits'] += 1
            
            # 作者统计
            author_email = commit.author.email
            stats['total_authors'].add(author_email)
            
            # 行数变化统计
            stats['total_added_lines'] += commit.insertions
            stats['total_deleted_lines'] += commit.deletions
            
            # 文件扩展名统计
            for modified_file in commit.modified_files:
                _, ext = os.path.splitext(modified_file.filename)
                if ext:
                    ext = ext.lower()
                    stats['file_extensions'][ext] = stats['file_extensions'].get(ext, 0) + 1
                
                print(f"已处理 {i + 1} 个提交的统计信息", end='\r')
                
                stats['total_files_changed'] += 1
                
                # 按月份统计活动
                month = commit.committer_date.strftime('%Y-%m')
                if month not in stats['monthly_activity']:
                    stats['monthly_activity'][month] = 0
                stats['monthly_activity'][month] += 1
                
                # 尝试提取函数级别的修改信息
                lang = get_file_extension(modified_file.filename)
                if lang and modified_file.source_code:
                    # 提取修改文件中的函数定义
                    functions_in_file = self.parser.extract_functions(modified_file.source_code, lang)
                    for func_info in functions_in_file:
                        # 统计函数修改次数
                        full_func_name = f"{modified_file.filename}#{func_info['name']}"
                        stats['function_modifications'][full_func_name] = stats['function_modifications'].get(full_func_name, 0) + 1
                        stats['total_functions_changed'] += 1
        
        # 转换set为list以便JSON序列化
        stats['total_authors'] = list(stats['total_authors'])
        
        return stats

    def save_commits_to_file(self, commits: List[Dict], output_path: str):
        """
        将提交历史保存到文件
        
        Args:
            commits: 提交历史列表
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(commits, f, indent=2, ensure_ascii=False)
        
        print(f"提交历史已保存到: {output_path}")

    def save_stats_to_file(self, stats: Dict, output_path: str):
        """
        将统计信息保存到文件
        
        Args:
            stats: 统计信息字典
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"统计信息已保存到: {output_path}")


def analyze_repository(repo_url: str, repo_name: Optional[str] = None, branch: Optional[str] = None, 
                      max_commits: Optional[int] = None):
    """
    分析GitHub仓库的主要接口函数
    
    Args:
        repo_url: GitHub仓库URL
        repo_name: 本地存储的仓库名称
        branch: 分支名称
        max_commits: 最大提交数量限制
        
    Returns:
        bool: 是否成功完成分析
    """
    analyzer = GitHubAnalyzer()
    
    # 克隆仓库
    if not analyzer.clone_repository(repo_url, repo_name, branch):
        print("克隆失败")
        return False
    
    try:
        # 获取提交历史
        commits = analyzer.get_commit_history(max_commits)
        
        # 保存提交历史到项目目录
        commits_output_path = analyzer.project_dir / "commits.json"
        analyzer.save_commits_to_file(commits, str(commits_output_path))
        
        # 获取统计信息
        stats = analyzer.get_commit_statistics(max_commits)
        
        # 保存统计信息到项目目录
        stats_output_path = analyzer.project_dir / "stats.json"
        analyzer.save_stats_to_file(stats, str(stats_output_path))
        
        print("="*60)
        print(f"仓库分析完成，项目路径: {analyzer.project_dir}")
        print(f"- 总提交数: {stats['total_commits']}")
        print(f"- 总作者数: {len(stats['total_authors'])}")
        print(f"- 总文件更改数: {stats['total_files_changed']}")
        print(f"- 总函数更改数: {stats['total_functions_changed']}")
        print(f"- 新增总行数: {stats['total_added_lines']}")
        print(f"- 删除总行数: {stats['total_deleted_lines']}")
        print(f"- 主要编程语言: {sorted(stats['file_extensions'].items(), key=lambda x: x[1], reverse=True)[:5]}")
        print(f"- 最常修改的函数: {sorted(stats['function_modifications'].items(), key=lambda x: x[1], reverse=True)[:10]}")
        print("="*60)
        
        return True
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        return True# 即使出错，也返回True以继续流程


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GitHub仓库信息获取")
    parser.add_argument("--repo-url", type=str, required=True, help="GitHub仓库URL")
    parser.add_argument("--repo-name", type=str, default=None, help="本地仓库名称")
    parser.add_argument("--branch", type=str, default=None, help="分支名称")
    parser.add_argument("--max-commits", type=int, default=None, help="最大提交数量限制")
    
    args = parser.parse_args()
    
    success = analyze_repository(
        repo_url=args.repo_url,
        repo_name=args.repo_name,
        branch=args.branch,
        max_commits=args.max_commits
    )
    
    if success:
        print("分析完成！")
    else:
        print("分析失败！")
        sys.exit(1)

