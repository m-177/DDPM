import sys

with open('train.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

try_line = -1
try_indent = 0
for i, line in enumerate(lines):
    if line.strip().startswith('try:'):
        try_line = i
        try_indent = len(line) - len(line.lstrip())
        print(f"Found try at line {i+1}, indent: {try_indent} spaces")
        break

if try_line >= 0:
    # 检查try后面是否有except或finally
    found_except = False
    found_finally = False
    for i in range(try_line + 1, min(try_line + 1000, len(lines))):
        stripped = lines[i].strip()
        if stripped.startswith('except'):
            print(f"Found except at line {i+1}: {stripped[:50]}")
            found_except = True
        elif stripped.startswith('finally:'):
            print(f"Found finally at line {i+1}: {stripped[:50]}")
            found_finally = True
        elif stripped.startswith('plt.ioff()'):
            print(f"Found plt.ioff() at line {i+1}")
            break
    
    if not found_except and not found_finally:
        print("ERROR: try statement missing except/finally clauses")
    elif found_except and not found_finally:
        print("WARNING: try has except but no finally")
    else:
        print("try-except-finally structure appears complete")
else:
    print("No try statement found")
