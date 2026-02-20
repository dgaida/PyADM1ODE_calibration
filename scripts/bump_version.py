import re
import os
from datetime import datetime

def bump_version():
    # 1. Read current version from pyproject.toml
    with open('pyproject.toml', 'r') as f:
        content = f.read()

    match = re.search(r'version = "(.*?)"', content)
    if not match:
        print("Could not find version in pyproject.toml")
        return

    current_version = match.group(1)
    parts = current_version.split('.')
    if len(parts) != 3:
        print(f"Invalid version format: {current_version}")
        return

    new_version = f"{parts[0]}.{parts[1]}.{int(parts[2]) + 1}"
    print(f"Bumping version from {current_version} to {new_version}")

    # 2. Update pyproject.toml
    new_content = re.sub(r'version = ".*?"', f'version = "{new_version}"', content, count=1)
    with open('pyproject.toml', 'w') as f:
        f.write(new_content)

    # 3. Update __init__.py files
    files_to_update = [
        'pyadm1ode_calibration/__init__.py',
        'tests/__init__.py',
        'tests/unit/__init__.py'
    ]

    for filepath in files_to_update:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                c = f.read()
            new_c = re.sub(r'__version__ = ".*?"', f'__version__ = "{new_version}"', c)
            with open(filepath, 'w') as f:
                f.write(new_c)
        else:
            print(f"Warning: {filepath} not found")

    # 4. Update CHANGELOG.md
    if os.path.exists('CHANGELOG.md'):
        with open('CHANGELOG.md', 'r') as f:
            lines = f.readlines()

        today = datetime.now().strftime('%Y-%m-%d')
        new_entry = f"\n## [{new_version}] - {today}\n\n### Miscellaneous Tasks\n- Version bump to {new_version}\n"

        # Find the line after the header
        header_end_index = 0
        for i, line in enumerate(lines):
            if line.startswith('# Changelog'):
                header_end_index = i + 1
                # Skip the next line if it's "All notable changes..."
                if i + 1 < len(lines) and "notable changes" in lines[i+1]:
                    header_end_index = i + 2
                break

        lines.insert(header_end_index, new_entry)

        with open('CHANGELOG.md', 'w') as f:
            f.writelines(lines)
    else:
        print("Warning: CHANGELOG.md not found")

if __name__ == "__main__":
    bump_version()
