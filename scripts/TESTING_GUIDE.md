# 🧪 EasyML Pipeline Testing Guide

This guide provides comprehensive testing strategies for your complete EasyML MLOps pipeline. Test everything from individual components to full end-to-end workflows.

## 🚀 Quick Start Testing

### 1. **Basic Health Check** (30 seconds)
```bash
# Test if server is running
curl http://localhost:8000/health

# Test API documentation
curl http://localhost:8000/docs
```

### 2. **Quick Pipeline Test** (2-3 minutes)
```bash
# Run essential tests only
./test_pipeline.sh --quick
```

### 3. **Complete Pipeline Test** (5-10 minutes)
```bash
# Run full end-to-end test
python3 test_complete_pipeline.py
```

---

## 📋 Testing Components

### 🔄 **Complete Pipeline Testing**

Test the entire ML pipeline from data upload to deployment:

```bash
# Full automated pipeline test
python3 test_complete_pipeline.py

# What it tests:
# ✅ Server health
# ✅ User registration & authentication  
# ✅ Project creation
# ✅ Data upload (synthetic dataset)
# ✅ Model training (Random Forest)
# ✅ Model versioning (DVC integration)
# ✅ Deployment preparation
# ✅ Prediction testing
# ✅ Database integration
```

**Expected Output:**
```
🚀 EasyML Complete Pipeline Test
============================================

[12:34:56] HEALTH: Testing server health...
[12:34:56] HEALTH: ✅ Server is running
[12:34:57] AUTH: Registering test user...
[12:34:58] AUTH: ✅ User registered successfully
[12:34:59] AUTH: ✅ User logged in successfully
...
📊 PIPELINE TEST SUMMARY
Tests Passed: 8/9 (88.9%)
🎉 PIPELINE TEST SUCCESSFUL!
```

---

### 🏗️ **Infrastructure Testing**

Test all infrastructure components:

```bash
# Complete infrastructure test
./test_pipeline.sh --full

# Quick infrastructure test  
./test_pipeline.sh --quick

# API endpoints only
./test_pipeline.sh --api-only
```

**Components Tested:**
- ✅ **Database Connections** (PostgreSQL + MongoDB)
- ✅ **DVC Setup** (Data Version Control)
- ✅ **Docker Configuration** 
- ✅ **API Endpoints**
- ✅ **GitHub Workflows**
- ✅ **Performance Metrics**
- ✅ **Security Basics**

---

### ⚡ **Performance & Load Testing**

Test system performance under load:

```bash
# Run load test with concurrent users
python3 test_load_performance.py

# Configuration (edit script to modify):
NUM_CONCURRENT_USERS = 10
NUM_REQUESTS_PER_USER = 5
```

**Load Test Scenarios:**
- 👥 **Concurrent Users**: Simulates multiple users
- 🔄 **Complete Journeys**: Auth → Project → Training
- 📊 **Performance Metrics**: Response times, throughput
- 🎯 **Success Rates**: Error rates and reliability

**Expected Output:**
```
⚡ EasyML Load Testing Suite
========================================
👤 Starting user 0 journey...
👤 Starting user 1 journey...
...
📊 LOAD TEST ANALYSIS
==================================================
Total Requests: 200
Successful Requests: 195
Success Rate: 97.50%

Response Time Statistics:
  Mean: 0.245s
  95th Percentile: 0.890s
  Max: 1.234s

🎯 Performance Assessment:
✅ EXCELLENT - System performs well under load
```

---

### ⚙️ **GitHub Actions Testing**

Validate GitHub workflows locally:

```bash
# Validate workflow syntax and configuration
python3 test_github_workflows.py

# Check specific workflow
act --workflow ci-cd-pipeline.yml --dry-run  # (requires 'act' tool)
```

**Workflow Validation:**
- ✅ **YAML Syntax** validation
- ✅ **Required Secrets** identification
- ✅ **Environment Variables** check
- ✅ **Docker Integration** verification
- ✅ **Azure Integration** validation

---

## 🧪 Manual Testing Scenarios

### **Scenario 1: Data Scientist Workflow**
1. Register new user
2. Create classification project
3. Upload CSV dataset
4. Train multiple models
5. Compare model performance
6. Deploy best model

### **Scenario 2: MLOps Engineer Workflow**
1. Monitor model performance
2. Check DVC model versions
3. Test deployment endpoints
4. Validate monitoring dashboards
5. Test rollback procedures

### **Scenario 3: Production Readiness**
1. Load test with realistic data
2. Security vulnerability scan
3. Database performance test
4. Container deployment test
5. Azure integration test

---

## 📊 Testing Environments

### **Development Environment**
```bash
# Local testing with Docker Compose
docker-compose up -d
./test_pipeline.sh --full
```

### **Staging Environment**
```bash
# Test against staging deployment
BASE_URL="https://easyml-staging.azurecontainerinstances.io" \
python3 test_complete_pipeline.py
```

### **Production Readiness**
```bash
# Full production readiness check
./test_pipeline.sh --full
python3 test_load_performance.py
python3 test_github_workflows.py
```

---

## 🔧 Troubleshooting Common Issues

### **Server Not Starting**
```bash
# Check logs
docker-compose logs app

# Check ports
netstat -tulpn | grep 8000

# Restart services
docker-compose down && docker-compose up -d
```

### **Database Connection Issues**
```bash
# Test database directly
python3 check_db_records.py

# Check environment variables
cat .env | grep -E "(POSTGRES|MONGO)"

# Test MongoDB SSL
python3 scripts/test_mongodb_ssl.py
```

### **DVC Issues**
```bash
# Check DVC status
dvc status

# Check DVC remote
dvc remote list

# Test DVC storage
ls -la dvc_storage/models/
```

### **Authentication Failures**
```bash
# Check JWT configuration
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"test","email":"test@test.com","password":"testpass123"}'
```

---

## 📈 Performance Benchmarks

### **Expected Performance**
- ⚡ **Health Check**: < 100ms
- 👤 **Authentication**: < 500ms  
- 📊 **Model Training**: < 30s (small dataset)
- 🚀 **Prediction**: < 200ms
- 📂 **File Upload**: < 2s (10MB file)

### **Load Test Targets**
- 🎯 **Success Rate**: > 95%
- ⏱️ **Response Time**: < 2s (95th percentile)
- 🔄 **Throughput**: > 10 requests/second
- 👥 **Concurrent Users**: 50+ users

---

## 🚨 Monitoring & Alerts

### **Health Monitoring**
```bash
# Continuous health monitoring
while true; do
  curl -f http://localhost:8000/health || echo "⚠️ Service down!"
  sleep 30
done
```

### **Performance Monitoring**
```bash
# Log response times
curl -w "Response time: %{time_total}s\n" http://localhost:8000/health
```

---

## 🎯 CI/CD Integration

### **Pre-commit Testing**
```bash
# Add to .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pipeline-test
        name: Pipeline Test
        entry: ./test_pipeline.sh --quick
        language: system
```

### **GitHub Actions Integration**
```yaml
# .github/workflows/test.yml
- name: Run Pipeline Tests
  run: |
    ./test_pipeline.sh --full
    python3 test_complete_pipeline.py
```

---

## 📝 Test Reports

All testing scripts generate detailed reports:

- 📊 **Performance Metrics**: Response times, throughput
- ✅ **Success Rates**: Pass/fail statistics  
- 🐛 **Error Analysis**: Detailed error breakdown
- 📈 **Trends**: Performance over time
- 🎯 **Recommendations**: Optimization suggestions

---

## 🚀 Next Steps

1. **Automated Testing**: Set up CI/CD pipeline testing
2. **Monitoring**: Implement production monitoring
3. **Scaling**: Test horizontal scaling capabilities
4. **Security**: Run security penetration testing
5. **Documentation**: Update API documentation

---

## ❓ Support & Resources

- 📚 **API Documentation**: `http://localhost:8000/docs`
- 🐛 **Issue Tracking**: GitHub Issues
- 📖 **User Guide**: `README.md`
- 🏗️ **Architecture**: `ARCHITECTURE.md`

---

**Testing is crucial for MLOps success! 🎯**

Regular testing ensures your pipeline remains reliable, performant, and ready for production workloads.
