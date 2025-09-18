#!/usr/bin/env python3
"""
Load Testing Script for EasyML API
Tests performance under load and concurrent users
"""
import asyncio
import aiohttp
import time
import json
import statistics
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Configuration
BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"
NUM_CONCURRENT_USERS = 10
NUM_REQUESTS_PER_USER = 5
TIMEOUT = 30

class LoadTester:
    def __init__(self):
        self.base_url = f"{BASE_URL}{API_PREFIX}"
        self.results = []
        
    async def create_session_with_auth(self):
        """Create authenticated session"""
        async with aiohttp.ClientSession() as session:
            # Register user
            timestamp = int(time.time() * 1000)
            user_data = {
                "username": f"loadtest_{timestamp}",
                "email": f"loadtest_{timestamp}@example.com",
                "password": "testpass123",
                "full_name": f"Load Test User {timestamp}"
            }
            
            try:
                async with session.post(f"{self.base_url}/auth/register", json=user_data) as resp:
                    if resp.status != 201:
                        return None
                
                # Login
                login_data = {
                    "username": user_data["username"],
                    "password": user_data["password"]
                }
                
                async with session.post(f"{self.base_url}/auth/login", data=login_data) as resp:
                    if resp.status != 200:
                        return None
                    
                    token_data = await resp.json()
                    return token_data["access_token"]
                    
            except Exception as e:
                print(f"Auth error: {e}")
                return None
    
    async def test_endpoint_performance(self, endpoint, method="GET", data=None, auth_token=None):
        """Test single endpoint performance"""
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as session:
                if method == "GET":
                    async with session.get(f"{BASE_URL}{endpoint}", headers=headers) as resp:
                        status = resp.status
                        response_time = time.time() - start_time
                        content_length = len(await resp.text())
                        
                elif method == "POST":
                    async with session.post(f"{BASE_URL}{endpoint}", json=data, headers=headers) as resp:
                        status = resp.status
                        response_time = time.time() - start_time
                        content_length = len(await resp.text())
                
                return {
                    "endpoint": endpoint,
                    "method": method,
                    "status": status,
                    "response_time": response_time,
                    "content_length": content_length,
                    "success": 200 <= status < 400,
                    "timestamp": datetime.now()
                }
                
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "endpoint": endpoint,
                "method": method,
                "status": 0,
                "response_time": response_time,
                "content_length": 0,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    async def simulate_user_journey(self, user_id):
        """Simulate complete user journey"""
        print(f"👤 Starting user {user_id} journey...")
        
        # Create authenticated session
        auth_token = await self.create_session_with_auth()
        if not auth_token:
            print(f"❌ User {user_id} authentication failed")
            return []
        
        user_results = []
        
        # Test endpoints in realistic order
        test_scenarios = [
            # Health check
            ("/health", "GET", None),
            
            # Get user info
            (f"{API_PREFIX}/auth/me", "GET", None),
            
            # Create project
            (f"{API_PREFIX}/projects/", "POST", {
                "name": f"Load Test Project {user_id}",
                "description": "Load testing project",
                "project_type": "classification"
            }),
            
            # List projects
            (f"{API_PREFIX}/projects/", "GET", None),
        ]
        
        for endpoint, method, data in test_scenarios:
            for request_num in range(NUM_REQUESTS_PER_USER):
                result = await self.test_endpoint_performance(
                    endpoint, method, data, auth_token
                )
                result["user_id"] = user_id
                result["request_num"] = request_num
                user_results.append(result)
                
                # Small delay between requests
                await asyncio.sleep(0.1)
        
        print(f"✅ User {user_id} completed journey")
        return user_results
    
    async def run_load_test(self):
        """Run complete load test"""
        print(f"🚀 Starting load test with {NUM_CONCURRENT_USERS} concurrent users")
        print(f"Each user will make {NUM_REQUESTS_PER_USER} requests per endpoint")
        
        start_time = time.time()
        
        # Create tasks for concurrent users
        tasks = []
        for user_id in range(NUM_CONCURRENT_USERS):
            task = asyncio.create_task(self.simulate_user_journey(user_id))
            tasks.append(task)
        
        # Wait for all users to complete
        all_results = await asyncio.gather(*tasks)
        
        # Flatten results
        for user_results in all_results:
            self.results.extend(user_results)
        
        total_time = time.time() - start_time
        
        print(f"⏱️ Load test completed in {total_time:.2f} seconds")
        
        return self.analyze_results()
    
    def analyze_results(self):
        """Analyze load test results"""
        if not self.results:
            print("❌ No results to analyze")
            return {}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.results)
        
        print("\n📊 LOAD TEST ANALYSIS")
        print("=" * 50)
        
        # Overall statistics
        total_requests = len(df)
        successful_requests = len(df[df['success'] == True])
        success_rate = (successful_requests / total_requests) * 100
        
        print(f"Total Requests: {total_requests}")
        print(f"Successful Requests: {successful_requests}")
        print(f"Success Rate: {success_rate:.2f}%")
        
        # Response time statistics
        response_times = df['response_time'].values
        print(f"\nResponse Time Statistics:")
        print(f"  Mean: {np.mean(response_times):.3f}s")
        print(f"  Median: {np.median(response_times):.3f}s")
        print(f"  95th Percentile: {np.percentile(response_times, 95):.3f}s")
        print(f"  99th Percentile: {np.percentile(response_times, 99):.3f}s")
        print(f"  Max: {np.max(response_times):.3f}s")
        print(f"  Min: {np.min(response_times):.3f}s")
        
        # Requests per second
        time_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
        rps = total_requests / time_span if time_span > 0 else 0
        print(f"\nThroughput: {rps:.2f} requests/second")
        
        # Per-endpoint analysis
        print(f"\nPer-Endpoint Analysis:")
        endpoint_stats = df.groupby('endpoint').agg({
            'response_time': ['mean', 'max', 'min'],
            'success': 'mean',
            'status': 'count'
        }).round(3)
        
        for endpoint in endpoint_stats.index:
            stats = endpoint_stats.loc[endpoint]
            print(f"  {endpoint}:")
            print(f"    Avg Response: {stats[('response_time', 'mean')]:.3f}s")
            print(f"    Success Rate: {stats[('success', 'mean')] * 100:.1f}%")
            print(f"    Total Requests: {stats[('status', 'count')]}")
        
        # Error analysis
        errors = df[df['success'] == False]
        if not errors.empty:
            print(f"\nError Analysis:")
            error_counts = errors['status'].value_counts()
            for status, count in error_counts.items():
                print(f"  HTTP {status}: {count} occurrences")
        
        # Performance assessment
        print(f"\n🎯 Performance Assessment:")
        if success_rate >= 95 and np.mean(response_times) < 2.0:
            print("✅ EXCELLENT - System performs well under load")
        elif success_rate >= 90 and np.mean(response_times) < 5.0:
            print("✅ GOOD - System handles load adequately")
        elif success_rate >= 80:
            print("⚠️ ACCEPTABLE - System shows some strain under load")
        else:
            print("❌ POOR - System struggles under load")
        
        return {
            "total_requests": total_requests,
            "success_rate": success_rate,
            "mean_response_time": np.mean(response_times),
            "p95_response_time": np.percentile(response_times, 95),
            "requests_per_second": rps,
            "assessment": "PASS" if success_rate >= 90 and np.mean(response_times) < 5.0 else "FAIL"
        }

async def test_simple_endpoints():
    """Test simple endpoints without authentication"""
    print("🔍 Testing simple endpoints...")
    
    simple_tests = [
        ("/health", "GET"),
        ("/docs", "GET"),
        ("/openapi.json", "GET")
    ]
    
    results = []
    
    async with aiohttp.ClientSession() as session:
        for endpoint, method in simple_tests:
            start_time = time.time()
            try:
                async with session.get(f"{BASE_URL}{endpoint}") as resp:
                    response_time = time.time() - start_time
                    success = 200 <= resp.status < 400
                    
                    print(f"  {endpoint}: {resp.status} ({response_time:.3f}s)")
                    results.append({
                        "endpoint": endpoint,
                        "success": success,
                        "response_time": response_time
                    })
                    
            except Exception as e:
                response_time = time.time() - start_time
                print(f"  {endpoint}: ERROR ({response_time:.3f}s) - {e}")
                results.append({
                    "endpoint": endpoint,
                    "success": False,
                    "response_time": response_time
                })
    
    return results

async def main():
    """Main test runner"""
    print("⚡ EasyML Load Testing Suite")
    print("=" * 40)
    
    # First test simple endpoints
    simple_results = await test_simple_endpoints()
    
    # Check if server is responsive
    if not any(r["success"] for r in simple_results):
        print("❌ Server is not responding. Cannot run load tests.")
        return
    
    # Run load test
    tester = LoadTester()
    load_results = await tester.run_load_test()
    
    print("\n" + "=" * 50)
    print("🏁 LOAD TESTING COMPLETED")
    print("=" * 50)
    
    if load_results.get("assessment") == "PASS":
        print("🎉 Load testing PASSED - System is ready for production load!")
    else:
        print("⚠️ Load testing FAILED - System needs optimization before production.")

if __name__ == "__main__":
    asyncio.run(main())
