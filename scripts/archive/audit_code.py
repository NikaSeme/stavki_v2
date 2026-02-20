import os
import ast
from pathlib import Path

def get_imports(file_path):
    with open(file_path, "r") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return set(), set()
            
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports

def check_project(root_dir):
    all_files = []
    for root, dirs, files in os.walk(root_dir):
        if "venv" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                all_files.append(os.path.join(root, file))

    print(f"Scanning {len(all_files)} files...")
    # This is a very basic check, just listing imports for now
    # A full unused import checker is complex (needs to check usage in file)
    # So instead, let's look for 'print' statements that might be left over debugging
    
    suspicious = []
    for file_path in all_files:
        with open(file_path, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "print(" in line and "scripts/" not in file_path:
                    suspicious.append(f"{file_path}:{i+1}: {line.strip()}")
                if "import pdb" in line or "breakpoint()" in line:
                    suspicious.append(f"{file_path}:{i+1}: DEBUGGER FOUND")
                    
    if suspicious:
        print("\n⚠️ Suspicious Code Found (prints/debuggers):")
        for s in suspicious:
            print(s)
    else:
        print("\n✅ No obvious debug prints found (excluding scripts/).")

if __name__ == "__main__":
    check_project("stavki")
