import os
import ast
import sys

def get_unused_imports(file_path):
    with open(file_path, "r") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return []
            
    imported = set()
    used = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                name = n.asname if n.asname else n.name
                imported.add(name)
        elif isinstance(node, ast.ImportFrom):
            for n in node.names:
                name = n.asname if n.asname else n.name
                imported.add(name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            # Checking attributes is harder, but simplified:
            # if we have "import os", used "os.path" counts as using "os"
            pass
            
    # Naive usage check
    unused = []
    for imp in imported:
        if imp not in used:
            # Double check common patterns
            is_used = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id == imp:
                    is_used = True
                    break
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == imp:
                    is_used = True
                    break
            
            if not is_used:
                unused.append(imp)
                
    return unused

def check_project(root_dir):
    for root, dirs, files in os.walk(root_dir):
        if "venv" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                unused = get_unused_imports(path)
                if unused:
                    print(f"{path}: Unused imports: {', '.join(unused)}")

if __name__ == "__main__":
    check_project("stavki")
