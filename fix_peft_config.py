#!/usr/bin/env python3
"""
Utility script to fix PEFT adapter configs that are missing required fields.
"""
import json
import sys
from pathlib import Path

def fix_peft_config(model_path: str):
    """Fix PEFT adapter config by ensuring required fields are present."""

    model_path = Path(model_path)
    adapter_config_path = model_path / "adapter_config.json"

    if not adapter_config_path.exists():
        print(f"❌ No adapter_config.json found in {model_path}")
        return False

    print(f"🔧 Checking PEFT config in {adapter_config_path}")

    # Load existing config
    with open(adapter_config_path, 'r') as f:
        config = json.load(f)

    print(f"📄 Current config: {json.dumps(config, indent=2)}")

    # Check for required fields
    required_fields = {
        'peft_type': 'LORA',
        'task_type': 'CAUSAL_LM'
    }

    changes_made = False
    for field, default_value in required_fields.items():
        if field not in config:
            print(f"➕ Adding missing field: {field} = {default_value}")
            config[field] = default_value
            changes_made = True
        else:
            print(f"✅ Field {field} already present: {config[field]}")

    if changes_made:
        # Backup original
        backup_path = adapter_config_path.with_suffix('.json.backup')
        print(f"💾 Creating backup: {backup_path}")
        with open(backup_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Save fixed config
        print(f"💾 Saving fixed config to {adapter_config_path}")
        with open(adapter_config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print("✅ PEFT config fixed!")
        return True
    else:
        print("✅ PEFT config is already correct!")
        return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python fix_peft_config.py <model_path>")
        print("Example: python fix_peft_config.py ./outputs")
        sys.exit(1)

    model_path = sys.argv[1]

    try:
        success = fix_peft_config(model_path)
        if success:
            print(f"\n🎉 PEFT config in {model_path} is now ready to load!")
        else:
            print(f"\n❌ Failed to fix PEFT config in {model_path}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()