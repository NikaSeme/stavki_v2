
import os
import ast
import sys
from pathlib import Path
from collections import defaultdict
import re

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def analyze_codebase():
    print("🔎 Starting Codebase Audit...")
    
    python_files = []
    for root, _, files in os.walk(PROJECT_ROOT):
        if "venv" in root or ".git" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                python_files.append(Path(root) / file)
                
    print(f"📄 Found {len(python_files)} Python files.")
    
    # 1. Performance Anti-patterns
    print("\n🚀 Performance Anti-patterns Audit:")
    performance_issues = []
    for file_path in python_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.splitlines()
                
            for i, line in enumerate(lines):
                if "iterrows" in line:
                    performance_issues.append((str(file_path.relative_to(PROJECT_ROOT)), i+1, "iterrows usage (slow loop)"))
                if ".apply(" in line and "axis=1" in line:
                    performance_issues.append((str(file_path.relative_to(PROJECT_ROOT)), i+1, "apply(axis=1) usage (slow loop)"))
                if "for index, row in df.iterrows():" in line:
                     performance_issues.append((str(file_path.relative_to(PROJECT_ROOT)), i+1, "Explicit iterrows loop"))

        except Exception as e:
            pass # Skip binary or irregular files

    if performance_issues:
        for issue in performance_issues[:10]: # Check top 10
            print(f"  ⚠️  {issue[0]}:{issue[1]} - {issue[2]}")
        if len(performance_issues) > 10:
            print(f"  ...and {len(performance_issues)-10} more.")
    else:
        print("  ✅ No obvious performance anti-patterns found.")

    # 2. Variable Consistency (Heuristic)
    print("\nHz Variable Consistency Check (Home Team):")
    home_team_vars = defaultdict(int)
    for file_path in python_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Regex for home team variable assignment or usage
            matches = re.findall(r'(home_team|HomeTeam|hometeam)', content)
            for m in matches:
                home_team_vars[m] += 1
        except: pass
        
    for var, count in home_team_vars.items():
        print(f"  - {var}: used {count} times")

    # 3. Large Files
    print("\n📦 Large Files (>500 lines):")
    for file_path in python_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = len(f.readlines())
            if lines > 500:
                print(f"  ⚠️  {file_path.name}: {lines} lines")
        except: pass

    # 4. Redundant Scripts?
    print("\n🗑️  Potential Redundant Scripts (Debug/Temp):")
    suspicious_keywords = ["debug", "test", "check", "verify", "inspect", "audit", "manual", "reproduce"]
    suspicious_files = []
    for file_path in python_files:
        if any(k in file_path.name.lower() for k in suspicious_keywords):
             suspicious_files.append(file_path.name)
    
    if suspicious_files:
        print(f"  Found {len(suspicious_files)} potential debug/maintenance scripts.")
        print(f"  Examples: {', '.join(suspicious_files[:5])}")

if __name__ == "__main__":
    analyze_codebase()
