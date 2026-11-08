import json
import ast

with open("Chest_XRay_Screening_Assistant_Explainability.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
all_valid = True
for i, cell in enumerate(nb['cells']):
    c_type = cell['cell_type']
    first_line = cell['source'][0].strip() if cell['source'] else ""
    print(f"Cell {i+1:02d} [{c_type:<8}]: {first_line[:65]}")
    if c_type == "code":
        code = "".join([line for line in cell['source'] if not line.strip().startswith("!")])
        try:
            ast.parse(code)
        except Exception as e:
            print(f"   ❌ Syntax error in cell {i+1}: {e}")
            all_valid = False

if all_valid:
    print("✅ All Python code cells in the notebook parsed with 0 syntax errors!")

