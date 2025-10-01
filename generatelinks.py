import re
import os

# Change to your actual repo URL
REPO_URL = "https://github.com/Vishvgehlot/Image-Classification-Model"

# Files and patterns to search
# (You can add as many (file, regex_pattern) pairs as you need)
FUNCTIONS = [
    ("test.ipynb", r"def\s+train_model"),
    ("test.ipynb", r"def\s+evaluate_model"),
]


README_PATH = "README.md"
OUTPUT_SECTION_HEADER = "## 📚 Code References"

def get_line_number(file_path, pattern):
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if re.search(pattern, line):
                return i
    return None

# Collect links
links = []
for file, pattern in FUNCTIONS:
    line_num = get_line_number(file, pattern)
    if line_num:
        link = f"{REPO_URL}/{file}#L{line_num}"
        links.append((pattern, link))
    else:
        print(f"⚠️ Pattern not found: {pattern} in {file}")

# Build markdown section
markdown_lines = [OUTPUT_SECTION_HEADER, "", "| Item | Link |", "|------|------|"]
for pattern, link in links:
    markdown_lines.append(f"| `{pattern}` | [View Code]({link}) |")

new_section = "\n".join(markdown_lines)

# Update README
with open(README_PATH, "r", encoding="utf-8") as f:
    content = f.read()

if OUTPUT_SECTION_HEADER in content:
    # Replace existing section
    content = re.sub(
        rf"{OUTPUT_SECTION_HEADER}[\s\S]*?(?=\n## |$)",
        new_section,
        content,
    )
else:
    # Append at the end
    content += "\n\n" + new_section

with open(README_PATH, "w", encoding="utf-8") as f:
    f.write(content)

print("✅ README.md updated with links!")
