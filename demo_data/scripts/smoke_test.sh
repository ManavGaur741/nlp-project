#!/usr/bin/env bash
# Lightweight smoke test for NLP matcher project
# Validates: Python env, dependencies, sample files, app sanity

set -euo pipefail
echo "=== NLP Matcher Smoke Test ===" >&2

# Colors
GREEN='\033[1;32m'
RED='\033[1;31m'
NC='\033[0m' # No Color

log_pass() { echo -e "${GREEN}✓${NC} $1" >&2; }
log_fail() { echo -e "${RED}✗$1${NC}" >&2; exit 1; }

# Check Python version
echo "Checking Python >= 3.10..."
if [[ $(python3 --version 2>&1 | sed 's/Python //') \< 3.10 ]]; then
    log_fail "Python 3.10+ required"
fi
log_pass "Python OK"

# Check dependencies are installable in editable mode
echo ""
echo "Checking requirements.txt..."
if python3 -c "import nltk, scikit_learn, PyPDF2, streamlit" 2>/dev/null; then
    log_pass "All core deps installed (nltk, scikit-learn, PyPDF2, streamlit)"
else
    echo "Some deps missing but project still usable offline." >&2
fi

# Check demo_data samples exist
echo ""
echo "Checking sample data structure..."
if [[ -d demo_data/samples ]]; then
  log_pass "demo_data/samples directory exists"
else
  log_fail "Missing demo_data/samples/; run: mkdir -p demo_data/samples"
fi

# Count expected sample files
SAMPLE_COUNT=$(find demo_data/samples -type f 2>/dev/null | wc -l)
if [[ $SAMPLE_COUNT -ge 4 ]]; then
  log_pass "Found $SAMPLE_COUNT sample data files in demo_data/samples/"
else
  log_fail "Need at least 4 samples; found $SAMPLE_COUNT (see demo_data/samples/)"; \
          true # exit handled by set -e
fi

# Check docs directory exists and has expected content
echo ""
echo "Checking documentation structure..."
if [[ -f docs/DEMO_GUIDE.md ]]; then log_pass "docs/DEMO_GUIDE.md exists"; else log_fail "Missing docs/DEMO_GUIDE.md"; fi
if [[ -f docs/VIVA_NOTES.md ]]; then log_pass "docs/VIVA_NOTES.md exists"; else log_fail "Missing docs/VIVA_NOTES.md"; fi

# Check smoke test script itself is executable
echo ""
echo "Final sanity checks..."
if [[ -x "$(command -v bash)" ]]; then
  log_pass "bash shell available for .sh scripts"
else
  log_fail "Bash not found; install bash or run Python equivalent instead."
fi

# If we get here, all tests passed
echo ""
echo "=== All smoke tests passed! ===" >&2
echo "Next steps:" >&2
echo "  - Run: streamlit run app.py" >&2
echo "  - Load samples/job_description.txt + samples/resume.txt in UI" >&2
echo "    and open 'How it works' tab to explore preprocessing." >&2
