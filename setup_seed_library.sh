#!/bin/bash
# Setup script to organize seed library files

echo "🚀 Popper RL - Seed Library Setup"
echo "=================================="

# Create data directory if it doesn't exist
mkdir -p /workspace/data

# Check for existing seed files in common locations
FILES_FOUND=0

# Look for the two specific files mentioned
for file in "adversarial_library_meta (1).jsonl" "adversarial_library_meta.jsonl" "vulnerability_library_featherless_adversarial.jsonl"; do
    # Search in current directory and subdirectories
    found=$(find /workspace -name "$file" -type f 2>/dev/null | head -1)
    
    if [ ! -z "$found" ]; then
        echo "✅ Found: $file at $found"
        
        # Copy to data directory
        cp "$found" "/workspace/data/"
        echo "   → Copied to /workspace/data/$file"
        FILES_FOUND=$((FILES_FOUND + 1))
    fi
done

echo ""
if [ $FILES_FOUND -gt 0 ]; then
    echo "✨ Setup complete! $FILES_FOUND file(s) organized."
    echo ""
    echo "Your seed libraries are now available at:"
    ls -lh /workspace/data/*.jsonl 2>/dev/null
    echo ""
    echo "Next steps:"
    echo "1. Set your GROQ_API_KEY environment variable"
    echo "2. Set your HF_TOKEN environment variable"
    echo "3. Run your backend: uvicorn backend.main:app --reload"
    echo "4. Start testing with dynamic prompt generation!"
else
    echo "⚠️ No seed library files found."
    echo ""
    echo "Please ensure these files are in your workspace:"
    echo "  - adversarial_library_meta (1).jsonl"
    echo "  - vulnerability_library_featherless_adversarial.jsonl"
    echo ""
    echo "You can place them in:"
    echo "  - /workspace/ (project root)"
    echo "  - /workspace/data/ (recommended)"
    echo "  - /workspace/scripts/"
fi
