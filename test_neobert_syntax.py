#!/usr/bin/env python3
"""Test neoBERT implementation syntax"""

import ast
import sys

def check_syntax(filepath):
    """Check Python file syntax"""
    with open(filepath, 'r') as f:
        try:
            ast.parse(f.read())
            return True, None
        except SyntaxError as e:
            return False, str(e)

def main():
    """Check syntax of all neoBERT-related files"""
    files_to_check = [
        'models/bert/config.py',
        'models/bert/core.py',
        'models/bert/core_base.py',
        'models/bert/layers/feedforward.py',
        'models/bert/layers/__init__.py',
        'models/bert/__init__.py',
    ]
    
    print("Checking neoBERT implementation syntax...")
    all_good = True
    
    for filepath in files_to_check:
        ok, error = check_syntax(filepath)
        if ok:
            print(f"✓ {filepath}")
        else:
            print(f"✗ {filepath}: {error}")
            all_good = False
    
    if all_good:
        print("\n✅ All neoBERT files have valid syntax!")
        
        # Also verify key functions exist
        print("\nVerifying neoBERT exports in __init__.py...")
        with open('models/bert/__init__.py', 'r') as f:
            content = f.read()
            
        exports = [
            'create_neobert',
            'create_neobert_mini',
            'create_neobert_core',
            'get_neobert_config',
            'get_neobert_mini_config',
            'NeoBertFeedForward',
            'SwiGLUMLP',
        ]
        
        for export in exports:
            if export in content:
                print(f"  ✓ Found {export}")
            else:
                print(f"  ✗ Missing {export}")
                all_good = False
        
        if all_good:
            print("\n✅ All neoBERT exports found!")
    else:
        print("\n❌ Some files have syntax errors")
        sys.exit(1)

if __name__ == "__main__":
    main()