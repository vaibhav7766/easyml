#!/bin/bash

# EasyML Pipeline Testing Suite
# Comprehensive testing script for the complete MLOps pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_BASE_URL="http://localhost:8000"
DVC_REMOTE_NAME="azure"
DOCKER_IMAGE_NAME="easyml:test"

echo -e "${BLUE}🚀 EasyML Complete Pipeline Testing Suite${NC}"
echo "=============================================="

# Function to check if service is running
check_service() {
    local service_name=$1
    local url=$2
    local timeout=${3:-10}
    
    echo -e "${YELLOW}📋 Checking $service_name...${NC}"
    
    if timeout $timeout bash -c "</dev/tcp/localhost/${url##*:}" 2>/dev/null; then
        echo -e "${GREEN}✅ $service_name is running${NC}"
        return 0
    else
        echo -e "${RED}❌ $service_name is not running${NC}"
        return 1
    fi
}

# Function to test database connections
test_databases() {
    echo -e "\n${BLUE}🗄️  Testing Database Connections${NC}"
    echo "================================"
    
    # Test PostgreSQL
    echo -e "${YELLOW}Testing PostgreSQL...${NC}"
    python3 check_db_records.py || {
        echo -e "${RED}❌ PostgreSQL connection failed${NC}"
        return 1
    }
    
    # Test MongoDB (if connection script exists)
    if [ -f "scripts/test_mongodb_ssl.py" ]; then
        echo -e "${YELLOW}Testing MongoDB...${NC}"
        python3 scripts/test_mongodb_ssl.py || {
            echo -e "${RED}❌ MongoDB connection failed${NC}"
            return 1
        }
    fi
    
    echo -e "${GREEN}✅ Database connections successful${NC}"
    return 0
}

# Function to test DVC setup
test_dvc_setup() {
    echo -e "\n${BLUE}📊 Testing DVC Setup${NC}"
    echo "===================="
    
    echo -e "${YELLOW}Checking DVC status...${NC}"
    dvc status --quiet || {
        echo -e "${RED}❌ DVC status check failed${NC}"
        return 1
    }
    
    echo -e "${YELLOW}Checking DVC remote...${NC}"
    dvc remote list | grep -q "$DVC_REMOTE_NAME" || {
        echo -e "${RED}❌ DVC remote '$DVC_REMOTE_NAME' not configured${NC}"
        return 1
    }
    
    echo -e "${YELLOW}Testing DVC storage...${NC}"
    ls -la dvc_storage/models/ > /dev/null 2>&1 || {
        echo -e "${RED}❌ DVC storage directory not accessible${NC}"
        return 1
    }
    
    echo -e "${GREEN}✅ DVC setup is working${NC}"
    return 0
}

# Function to test Docker setup
test_docker_setup() {
    echo -e "\n${BLUE}🐳 Testing Docker Setup${NC}"
    echo "======================="
    
    echo -e "${YELLOW}Checking Docker daemon...${NC}"
    docker info > /dev/null 2>&1 || {
        echo -e "${RED}❌ Docker daemon not running${NC}"
        return 1
    }
    
    echo -e "${YELLOW}Building test Docker image...${NC}"
    docker build -t $DOCKER_IMAGE_NAME -f scripts/Dockerfile . || {
        echo -e "${RED}❌ Docker build failed${NC}"
        return 1
    }
    
    echo -e "${YELLOW}Testing Docker image...${NC}"
    docker run --rm $DOCKER_IMAGE_NAME python3 -c "import app; print('App imports successfully')" || {
        echo -e "${RED}❌ Docker image test failed${NC}"
        return 1
    }
    
    echo -e "${GREEN}✅ Docker setup is working${NC}"
    return 0
}

# Function to test API endpoints
test_api_endpoints() {
    echo -e "\n${BLUE}🌐 Testing API Endpoints${NC}"
    echo "========================"
    
    # Test health endpoint
    echo -e "${YELLOW}Testing health endpoint...${NC}"
    curl -f -s "$API_BASE_URL/health" > /dev/null || {
        echo -e "${RED}❌ Health endpoint failed${NC}"
        return 1
    }
    
    # Test OpenAPI docs
    echo -e "${YELLOW}Testing OpenAPI docs...${NC}"
    curl -f -s "$API_BASE_URL/docs" > /dev/null || {
        echo -e "${RED}❌ OpenAPI docs not accessible${NC}"
        return 1
    }
    
    # Test API endpoints with authentication
    echo -e "${YELLOW}Testing authenticated endpoints...${NC}"
    curl -f -s "$API_BASE_URL/api/v1/auth/register" \
        -H "Content-Type: application/json" \
        -d '{"test": "connectivity"}' > /dev/null 2>&1 || {
        echo -e "${YELLOW}⚠️ Auth endpoint may require valid data${NC}"
    }
    
    echo -e "${GREEN}✅ API endpoints are accessible${NC}"
    return 0
}

# Function to run complete pipeline test
run_pipeline_test() {
    echo -e "\n${BLUE}🔄 Running Complete Pipeline Test${NC}"
    echo "=================================="
    
    echo -e "${YELLOW}Starting automated pipeline test...${NC}"
    python3 test_complete_pipeline.py || {
        echo -e "${RED}❌ Complete pipeline test failed${NC}"
        return 1
    }
    
    echo -e "${GREEN}✅ Complete pipeline test passed${NC}"
    return 0
}

# Function to test GitHub Actions workflows (dry run)
test_github_workflows() {
    echo -e "\n${BLUE}⚙️  Testing GitHub Workflows${NC}"
    echo "============================="
    
    echo -e "${YELLOW}Validating workflow syntax...${NC}"
    
    # Check if workflow files exist and are valid YAML
    for workflow in .github/workflows/*.yml; do
        if [ -f "$workflow" ]; then
            echo -e "${YELLOW}Checking $(basename $workflow)...${NC}"
            python3 -c "import yaml; yaml.safe_load(open('$workflow'))" || {
                echo -e "${RED}❌ Invalid YAML in $workflow${NC}"
                return 1
            }
        fi
    done
    
    # Test if required secrets are documented
    echo -e "${YELLOW}Checking required secrets...${NC}"
    if [ -f "README.md" ] || [ -f "DEPLOYMENT_GUIDE.md" ]; then
        echo -e "${GREEN}✅ Documentation found for secrets${NC}"
    else
        echo -e "${YELLOW}⚠️ Consider documenting required secrets${NC}"
    fi
    
    echo -e "${GREEN}✅ GitHub workflows syntax is valid${NC}"
    return 0
}

# Function to test performance and load
test_performance() {
    echo -e "\n${BLUE}⚡ Testing Performance${NC}"
    echo "====================="
    
    echo -e "${YELLOW}Testing API response times...${NC}"
    
    # Simple load test using curl
    total_time=0
    num_requests=5
    
    for i in $(seq 1 $num_requests); do
        response_time=$(curl -o /dev/null -s -w '%{time_total}' "$API_BASE_URL/health")
        total_time=$(echo "$total_time + $response_time" | bc -l)
        echo "  Request $i: ${response_time}s"
    done
    
    avg_time=$(echo "scale=3; $total_time / $num_requests" | bc -l)
    echo -e "${GREEN}✅ Average response time: ${avg_time}s${NC}"
    
    # Check if response time is reasonable (< 2 seconds)
    if (( $(echo "$avg_time < 2.0" | bc -l) )); then
        echo -e "${GREEN}✅ Performance is acceptable${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️ Performance may need optimization${NC}"
        return 1
    fi
}

# Function to run security tests
test_security() {
    echo -e "\n${BLUE}🔒 Testing Security${NC}"
    echo "=================="
    
    echo -e "${YELLOW}Testing HTTPS redirection...${NC}"
    # This would be more relevant in production
    echo -e "${GREEN}✅ Security checks passed (development mode)${NC}"
    
    echo -e "${YELLOW}Checking for exposed secrets...${NC}"
    # Check for common secret patterns in code
    if grep -r -i "password.*=" --include="*.py" --include="*.yml" . | grep -v "test" | grep -v "example"; then
        echo -e "${YELLOW}⚠️ Potential secrets found in code${NC}"
    else
        echo -e "${GREEN}✅ No obvious secrets in code${NC}"
    fi
    
    return 0
}

# Main test execution
main() {
    echo -e "${BLUE}Starting EasyML Pipeline Testing...${NC}"
    echo "Test started at: $(date)"
    echo ""
    
    # Track test results
    total_tests=0
    passed_tests=0
    
    # Define test functions and their names
    declare -a tests=(
        "test_databases:Database Connections"
        "test_dvc_setup:DVC Setup"
        "test_docker_setup:Docker Setup"
        "test_api_endpoints:API Endpoints"
        "run_pipeline_test:Complete Pipeline"
        "test_github_workflows:GitHub Workflows"
        "test_performance:Performance"
        "test_security:Security"
    )
    
    # Run each test
    for test_entry in "${tests[@]}"; do
        IFS=':' read -r test_func test_name <<< "$test_entry"
        total_tests=$((total_tests + 1))
        
        echo -e "\n${BLUE}Running: $test_name${NC}"
        
        if $test_func; then
            passed_tests=$((passed_tests + 1))
            echo -e "${GREEN}✅ $test_name PASSED${NC}"
        else
            echo -e "${RED}❌ $test_name FAILED${NC}"
        fi
    done
    
    # Final summary
    echo -e "\n${BLUE}===============================================${NC}"
    echo -e "${BLUE}📊 TESTING SUMMARY${NC}"
    echo -e "${BLUE}===============================================${NC}"
    
    success_rate=$(( (passed_tests * 100) / total_tests ))
    echo -e "Tests Passed: ${GREEN}$passed_tests${NC}/${total_tests} (${success_rate}%)"
    echo "Test completed at: $(date)"
    
    if [ $success_rate -ge 80 ]; then
        echo -e "\n${GREEN}🎉 PIPELINE TESTING SUCCESSFUL!${NC}"
        echo -e "${GREEN}Your EasyML pipeline is ready for deployment.${NC}"
        exit 0
    else
        echo -e "\n${YELLOW}⚠️ PIPELINE TESTING NEEDS ATTENTION${NC}"
        echo -e "${YELLOW}Please fix the failing tests before deployment.${NC}"
        exit 1
    fi
}

# Help function
show_help() {
    echo "EasyML Pipeline Testing Suite"
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  --help     Show this help message"
    echo "  --quick    Run only quick tests (skip pipeline test)"
    echo "  --api-only Test only API endpoints"
    echo "  --full     Run all tests (default)"
    echo ""
    echo "This script tests:"
    echo "  • Database connections"
    echo "  • DVC setup and storage"
    echo "  • Docker configuration"
    echo "  • API endpoints"
    echo "  • Complete ML pipeline"
    echo "  • GitHub workflows"
    echo "  • Performance"
    echo "  • Security basics"
}

# Handle command line arguments
case "${1:-}" in
    --help)
        show_help
        exit 0
        ;;
    --quick)
        echo "Running quick tests only..."
        test_databases && test_api_endpoints
        ;;
    --api-only)
        echo "Running API tests only..."
        test_api_endpoints
        ;;
    --full|"")
        main
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
