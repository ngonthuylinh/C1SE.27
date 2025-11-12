#!/usr/bin/env python3
"""
Test script cho Form Agent AI
"""

import asyncio
import json
import requests
import time
from datetime import datetime

# Configuration
API_BASE = "http://localhost:8000"
TEST_KEYWORDS = [
    "cloud security assessment",
    "investment portfolio analysis", 
    "digital marketing campaign planning",
    "machine learning model deployment",
    "financial risk management",
    "social media marketing strategy"
]

def test_api_health():
    """Test API health check"""
    print("🏥 Testing API Health...")
    
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ API is healthy: {data['status']}")
            print(f"   📊 Models loaded: {data['models_loaded']}")
            return True
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Connection failed: {str(e)}")
        return False

def test_form_generation():
    """Test form generation endpoint"""
    print("\n📝 Testing Form Generation...")
    
    results = []
    
    for keyword in TEST_KEYWORDS:
        print(f"\n   Testing keyword: '{keyword}'")
        
        start_time = time.time()
        
        try:
            response = requests.post(f"{API_BASE}/api/generate-form", 
                json={"keyword": keyword}
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Generated form: {data['title']}")
                print(f"   📂 Category: {data['category']}")
                print(f"   🔧 Complexity: {data['complexity']}")
                print(f"   📋 Fields: {len(data['fields'])}")
                print(f"   ⏱️  Time: {elapsed:.2f}s")
                
                results.append({
                    "keyword": keyword,
                    "success": True,
                    "form_id": data["form_id"],
                    "category": data["category"],
                    "complexity": data["complexity"],
                    "field_count": len(data["fields"]),
                    "response_time": elapsed
                })
            else:
                print(f"   ❌ Failed: {response.status_code}")
                results.append({
                    "keyword": keyword,
                    "success": False,
                    "error": response.text,
                    "response_time": elapsed
                })
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"   ❌ Exception: {str(e)}")
            results.append({
                "keyword": keyword,
                "success": False,
                "error": str(e),
                "response_time": elapsed
            })
    
    return results

def test_form_submission(form_id):
    """Test form submission"""
    print(f"\n📤 Testing Form Submission for {form_id}...")
    
    # Sample form data
    sample_data = {
        "full_name": "Test User",
        "email": "test@example.com",
        "phone": "+84987654321",
        "company": "Test Company"
    }
    
    try:
        response = requests.post(f"{API_BASE}/api/submit-form", 
            json={
                "form_id": form_id,
                "form_data": sample_data
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Submission successful: {data['submission_id']}")
            print(f"   📊 Status: {data['status']}")
            return data['submission_id']
        else:
            print(f"   ❌ Submission failed: {response.status_code}")
            print(f"   📄 Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
        return None

def test_analytics():
    """Test analytics endpoint"""
    print("\n📊 Testing Analytics...")
    
    try:
        response = requests.get(f"{API_BASE}/api/analytics")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Analytics retrieved")
            print(f"   📝 Total forms: {data['total_forms_generated']}")
            print(f"   📤 Total submissions: {data['total_submissions']}")
            print(f"   🔥 Popular keywords: {len(data['popular_keywords'])}")
            return True
        else:
            print(f"   ❌ Analytics failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
        return False

def test_load_performance():
    """Test load performance"""
    print("\n⚡ Testing Load Performance...")
    
    keyword = "performance test"
    num_requests = 10
    
    response_times = []
    
    for i in range(num_requests):
        start_time = time.time()
        
        try:
            response = requests.post(f"{API_BASE}/api/generate-form", 
                json={"keyword": f"{keyword} {i}"}
            )
            
            elapsed = time.time() - start_time
            response_times.append(elapsed)
            
            if response.status_code == 200:
                print(f"   ✅ Request {i+1}/{num_requests}: {elapsed:.2f}s")
            else:
                print(f"   ❌ Request {i+1}/{num_requests} failed: {response.status_code}")
                
        except Exception as e:
            elapsed = time.time() - start_time
            response_times.append(elapsed)
            print(f"   ❌ Request {i+1}/{num_requests} error: {str(e)}")
    
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        print(f"\n   📈 Performance Summary:")
        print(f"   ⏱️  Average: {avg_time:.2f}s")
        print(f"   🚀 Fastest: {min_time:.2f}s")
        print(f"   🐌 Slowest: {max_time:.2f}s")
        
        return {
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "total_requests": num_requests,
            "success_rate": len([t for t in response_times if t < 5.0]) / len(response_times)
        }
    
    return None

def generate_test_report(test_results):
    """Generate test report"""
    print("\n" + "="*60)
    print("📋 TEST REPORT")
    print("="*60)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "api_base": API_BASE,
        "test_results": test_results
    }
    
    # Save report
    report_filename = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"📄 Report saved: {report_filename}")
    
    # Summary
    if 'form_generation' in test_results:
        results = test_results['form_generation']
        successful = len([r for r in results if r['success']])
        total = len(results)
        success_rate = (successful / total) * 100 if total > 0 else 0
        
        print(f"\n📊 Form Generation Summary:")
        print(f"   ✅ Success rate: {success_rate:.1f}% ({successful}/{total})")
        
        if successful > 0:
            avg_time = sum([r['response_time'] for r in results if r['success']]) / successful
            print(f"   ⏱️  Average response time: {avg_time:.2f}s")
    
    if 'performance' in test_results:
        perf = test_results['performance']
        print(f"\n⚡ Performance Summary:")
        print(f"   📈 Average response time: {perf['avg_time']:.2f}s")
        print(f"   📊 Success rate: {perf['success_rate']*100:.1f}%")

def main():
    """Main test function"""
    print("🧪 Form Agent AI - Automated Testing")
    print("="*50)
    
    test_results = {}
    
    # Test 1: API Health
    if not test_api_health():
        print("\n❌ API not available. Please start the server first.")
        print("   Run: python main.py")
        return
    
    # Test 2: Form Generation
    form_results = test_form_generation()
    test_results['form_generation'] = form_results
    
    # Test 3: Form Submission (using first successful form)
    successful_forms = [r for r in form_results if r['success']]
    if successful_forms:
        form_id = successful_forms[0]['form_id']
        submission_id = test_form_submission(form_id)
        test_results['form_submission'] = {
            "form_id": form_id,
            "submission_id": submission_id,
            "success": submission_id is not None
        }
    
    # Test 4: Analytics
    analytics_success = test_analytics()
    test_results['analytics'] = {"success": analytics_success}
    
    # Test 5: Performance
    performance_results = test_load_performance()
    if performance_results:
        test_results['performance'] = performance_results
    
    # Generate report
    generate_test_report(test_results)
    
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    main()
