import json
import os
from typing import List, Dict, Any
from pathlib import Path

def load_jsonl_file(filepath: str) -> List[Dict[str, Any]]:
    """Load a single JSONL file."""
    data = []
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return data
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        # Normalize keys
                        if 'prompt' in entry:
                            data.append(entry)
                        elif 'text' in entry:
                            entry['prompt'] = entry['text']
                            data.append(entry)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    
    return data

def load_merged_library() -> List[Dict[str, Any]]:
    """
    Merges the two specific seed library files:
    1. adversarial_library_meta (1).jsonl
    2. vulnerability_library_featherless_adversarial.jsonl
    """
    base_dir = Path(__file__).parent.parent
    library_dir = base_dir / "data" # Assuming you put them in a data folder, or root
    
    # Try current directory first, then data subfolder
    search_paths = [
        Path("."),
        base_dir,
        base_dir / "data",
        base_dir / "scripts"
    ]
    
    all_prompts = []
    
    filenames = [
        "adversarial_library_meta (1).jsonl",
        "adversarial_library_meta.jsonl", # Fallback if space removed
        "vulnerability_library_featherless_adversarial.jsonl"
    ]
    
    found_count = 0
    
    for search_path in search_paths:
        for filename in filenames:
            filepath = search_path / filename
            if filepath.exists():
                data = load_jsonl_file(str(filepath))
                if data:
                    print(f"✅ Loaded {len(data)} prompts from {filename}")
                    all_prompts.extend(data)
                    found_count += 1
    
    if not all_prompts:
        print("⚠️ No seed libraries found. Ensure files are in the project root or 'data' folder.")
        print("Expected files:")
        print("  - adversarial_library_meta (1).jsonl")
        print("  - vulnerability_library_featherless_adversarial.jsonl")
        return []
    
    print(f"🚀 Total merged seed library: {len(all_prompts)} prompts")
    return all_prompts

# Global cache
PROMPT_LIBRARY = load_merged_library()

def get_prompts_by_weakness(weakness_type: str) -> List[str]:
    """Filter prompts by weakness type if available, else return all."""
    if not PROMPT_LIBRARY:
        return []
        
    filtered = [
        p.get('prompt', '') 
        for p in PROMPT_LIBRARY 
        if p.get('type', '').lower() == weakness_type.lower() or 
           p.get('weakness', '').lower() == weakness_type.lower() or
           p.get('category', '').lower() == weakness_type.lower()
    ]
    
    # If no specific matches, return a sample of all prompts to avoid empty sets
    if not filtered:
        return [p.get('prompt', '') for p in PROMPT_LIBRARY[:20]]
    
    return filtered

def get_random_seed() -> str:
    """Get a random seed prompt from the merged library."""
    if not PROMPT_LIBRARY:
        return "Generate an adversarial prompt."
    import random
    return random.choice(PROMPT_LIBRARY).get('prompt', 'Generate an adversarial prompt.')
