import os
import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

class DependencyAnalyzer:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.dependencies = {}
        self.module_files = []
        self.external_deps = set()
        
    def find_python_files(self) -> List[Path]:
        """查找所有Python文件"""
        python_files = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        return python_files
    
    def analyze_imports(self, file_path: Path) -> Tuple[Set[str], Set[str]]:
        """分析文件中的导入语句"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return set(), set()
        
        internal_imports = set()
        external_imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not alias.name.startswith('.'):
                        external_imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    if node.module.startswith('src.'):
                        # 内部导入
                        internal_imports.add(node.module)
                    else:
                        # 外部导入
                        external_imports.add(node.module.split('.')[0])
                
        return internal_imports, external_imports
    
    def analyze_project(self):
        """分析整个项目的依赖关系"""
        python_files = self.find_python_files()
        
        for file_path in python_files:
            # 跳过测试文件和__pycache__目录
            if 'test' in file_path.parts or '__pycache__' in file_path.parts:
                continue
                
            relative_path = file_path.relative_to(self.root_dir)
            module_name = str(relative_path).replace('.py', '').replace(os.sep, '.')
            
            internal_imports, external_imports = self.analyze_imports(file_path)
            
            self.dependencies[module_name] = {
                'file_path': str(relative_path),
                'internal_deps': list(internal_imports),
                'external_deps': list(external_imports)
            }
            
            self.external_deps.update(external_imports)
            
    def generate_dependency_graph(self) -> Dict:
        """生成依赖关系图"""
        graph = {
            'nodes': [],
            'edges': []
        }
        
        # 添加节点
        for module, info in self.dependencies.items():
            graph['nodes'].append({
                'id': module,
                'label': module,
                'file_path': info['file_path']
            })
        
        # 添加边
        for module, info in self.dependencies.items():
            for dep in info['internal_deps']:
                if dep in self.dependencies:
                    graph['edges'].append({
                        'from': module,
                        'to': dep
                    })
        
        return graph
    
    def save_results(self, output_dir: str):
        """保存分析结果"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 保存依赖关系数据
        with open(output_path / 'dependencies.json', 'w', encoding='utf-8') as f:
            json.dump(self.dependencies, f, indent=2, ensure_ascii=False)
        
        # 保存依赖关系图
        graph = self.generate_dependency_graph()
        with open(output_path / 'dependency_graph.json', 'w', encoding='utf-8') as f:
            json.dump(graph, f, indent=2, ensure_ascii=False)
        
        # 保存外部依赖
        with open(output_path / 'external_dependencies.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(self.external_deps)))
        
        # 生成模块依赖报告
        self.generate_report(output_path)
    
    def generate_report(self, output_path: Path):
        """生成依赖关系报告"""
        report_path = output_path / 'dependency_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('# 模块依赖关系报告\n\n')
            
            # 统计信息
            f.write('## 统计信息\n\n')
            f.write(f'- 总模块数: {len(self.dependencies)}\n')
            f.write(f'- 外部依赖数: {len(self.external_deps)}\n')
            f.write(f'- 平均依赖数: {sum(len(info["internal_deps"]) for info in self.dependencies.values()) / len(self.dependencies):.2f}\n\n')
            
            # 外部依赖列表
            f.write('## 外部依赖\n\n')
            for dep in sorted(self.external_deps):
                f.write(f'- {dep}\n')
            f.write('\n')
            
            # 模块依赖详情
            f.write('## 模块依赖详情\n\n')
            for module, info in sorted(self.dependencies.items()):
                f.write(f'### {module}\n\n')
                f.write(f'**文件路径**: `{info["file_path"]}`\n\n')
                
                if info['internal_deps']:
                    f.write('**内部依赖**:\n')
                    for dep in info['internal_deps']:
                        f.write(f'- {dep}\n')
                    f.write('\n')
                
                if info['external_deps']:
                    f.write('**外部依赖**:\n')
                    for dep in info['external_deps']:
                        f.write(f'- {dep}\n')
                    f.write('\n')

if __name__ == '__main__':
    analyzer = DependencyAnalyzer('src')
    analyzer.analyze_project()
    analyzer.save_results('reports/dependencies')
    print('依赖关系分析完成，结果已保存到 reports/dependencies 目录')