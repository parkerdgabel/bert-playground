#!/bin/bash
# Run the test suite with coverage

set -e  # Exit on error

echo "🧪 Running MLX BERT Playground Test Suite"
echo "========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse command line arguments
RUN_SLOW=false
RUN_INTEGRATION=false
SPECIFIC_TEST=""
COVERAGE_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --slow)
            RUN_SLOW=true
            shift
            ;;
        --integration)
            RUN_INTEGRATION=true
            shift
            ;;
        --all)
            RUN_SLOW=true
            RUN_INTEGRATION=true
            shift
            ;;
        --coverage)
            COVERAGE_ONLY=true
            shift
            ;;
        --test)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run_tests.sh [--slow] [--integration] [--all] [--coverage] [--test <test_name>]"
            exit 1
            ;;
    esac
done

# Clean up previous coverage data
echo -e "${YELLOW}Cleaning up previous test artifacts...${NC}"
rm -rf .coverage htmlcov .pytest_cache
rm -rf tests/__pycache__ tests/*/__pycache__

# Install test dependencies if needed
echo -e "${YELLOW}Installing test dependencies...${NC}"
uv pip install -e ".[dev]" --quiet

# Build pytest command
PYTEST_CMD="uv run pytest"

# Add coverage if not running specific test
if [ -z "$SPECIFIC_TEST" ] && [ "$COVERAGE_ONLY" != true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=. --cov-report=html --cov-report=term-missing"
fi

# Add markers based on flags
if [ "$RUN_SLOW" = false ] && [ "$RUN_INTEGRATION" = false ]; then
    echo -e "${GREEN}Running unit tests only (fast)...${NC}"
    PYTEST_CMD="$PYTEST_CMD -m 'not slow and not integration'"
elif [ "$RUN_SLOW" = true ] && [ "$RUN_INTEGRATION" = false ]; then
    echo -e "${GREEN}Running unit and slow tests...${NC}"
    PYTEST_CMD="$PYTEST_CMD -m 'not integration'"
elif [ "$RUN_SLOW" = false ] && [ "$RUN_INTEGRATION" = true ]; then
    echo -e "${GREEN}Running unit and integration tests...${NC}"
    PYTEST_CMD="$PYTEST_CMD -m 'not slow'"
else
    echo -e "${GREEN}Running all tests (including slow and integration)...${NC}"
fi

# Add specific test if provided
if [ -n "$SPECIFIC_TEST" ]; then
    echo -e "${GREEN}Running specific test: $SPECIFIC_TEST${NC}"
    PYTEST_CMD="$PYTEST_CMD $SPECIFIC_TEST -vv"
fi

# Add other pytest options
PYTEST_CMD="$PYTEST_CMD -vv --tb=short"

# Run tests
echo -e "${YELLOW}Command: $PYTEST_CMD${NC}"
echo "========================================="

if $PYTEST_CMD; then
    echo -e "\n${GREEN}✅ All tests passed!${NC}"
    
    # Show coverage report location if coverage was run
    if [ -z "$SPECIFIC_TEST" ] && [ "$COVERAGE_ONLY" != true ]; then
        echo -e "\n${YELLOW}📊 Coverage report generated:${NC}"
        echo "   - Terminal report above"
        echo "   - HTML report: htmlcov/index.html"
        echo -e "\n${YELLOW}To view HTML coverage report:${NC}"
        echo "   open htmlcov/index.html  # macOS"
        echo "   xdg-open htmlcov/index.html  # Linux"
    fi
else
    echo -e "\n${RED}❌ Some tests failed!${NC}"
    exit 1
fi

# If coverage only mode, generate coverage report
if [ "$COVERAGE_ONLY" = true ]; then
    echo -e "\n${YELLOW}Generating coverage report...${NC}"
    uv run coverage run -m pytest -m "not slow and not integration"
    uv run coverage report
    uv run coverage html
    echo -e "\n${GREEN}Coverage report generated in htmlcov/index.html${NC}"
fi

# Run linting checks
echo -e "\n${YELLOW}Running linting checks...${NC}"
echo "========================================="

# Run ruff
if uv run ruff check . --exclude=legacy; then
    echo -e "${GREEN}✅ Ruff linting passed${NC}"
else
    echo -e "${RED}❌ Ruff linting failed${NC}"
    exit 1
fi

# Run black check
if uv run black --check . --exclude=legacy; then
    echo -e "${GREEN}✅ Black formatting check passed${NC}"
else
    echo -e "${RED}❌ Black formatting check failed${NC}"
    echo -e "${YELLOW}Run 'black .' to fix formatting${NC}"
    exit 1
fi

echo -e "\n${GREEN}🎉 All checks passed!${NC}"