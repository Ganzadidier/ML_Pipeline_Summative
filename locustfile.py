from locust import HttpUser, task, between, events
import os
import random
import time
from io import BytesIO


class MLPredictionUser(HttpUser):
    """
    Locust user class for load testing ML prediction API
    """
    # Wait time between tasks (1-3 seconds)
    wait_time = between(1, 3)

    def on_start(self):
        """
        Called when a user starts - can be used for login, setup, etc.
        """
        # Create a dummy image file for testing
        self.test_image_data = self.create_test_image()

    def create_test_image(self):
        """
        Create a test image in memory for predictions
        """
        from PIL import Image
        import numpy as np

        # Create a random grayscale image (224x224)
        img_array = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')

        # Convert to RGB
        img_rgb = img.convert('RGB')

        # Save to bytes
        img_bytes = BytesIO()
        img_rgb.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        return img_bytes.getvalue()

    @task(10)
    def predict_single_image(self):
        """
        Test single image prediction endpoint (most common operation)
        Weight: 10 (most frequent task)
        """
        files = {'file': ('test_xray.jpg', BytesIO(self.test_image_data), 'image/jpeg')}

        with self.client.post(
                "/predict",
                files=files,
                catch_response=True,
                name="POST /predict"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'prediction' in data and 'confidence' in data:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except Exception as e:
                    response.failure(f"JSON parse error: {e}")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(5)
    def get_metrics(self):
        """
        Test metrics endpoint
        Weight: 5
        """
        with self.client.get("/metrics", catch_response=True, name="GET /metrics") as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'uptime_seconds' in data:
                        response.success()
                    else:
                        response.failure("Invalid metrics format")
                except:
                    response.failure("JSON parse error")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(3)
    def get_visualization_data(self):
        """
        Test visualization data endpoint
        Weight: 3
        """
        with self.client.get(
                "/visualization-data",
                catch_response=True,
                name="GET /visualization-data"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(2)
    def health_check(self):
        """
        Test health check endpoint
        Weight: 2
        """
        with self.client.get("/health", catch_response=True, name="GET /health") as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get('status') == 'healthy':
                        response.success()
                    else:
                        response.failure("Unhealthy status")
                except:
                    response.failure("JSON parse error")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(1)
    def get_predictions_log(self):
        """
        Test predictions log endpoint
        Weight: 1
        """
        with self.client.get(
                "/predictions-log",
                catch_response=True,
                name="GET /predictions-log"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


class StressTestUser(HttpUser):
    """
    Aggressive stress testing user - fires requests rapidly
    """
    wait_time = between(0.1, 0.5)  # Very short wait time

    def on_start(self):
        from PIL import Image
        import numpy as np

        img_array = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L').convert('RGB')

        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        self.test_image_data = img_bytes.getvalue()

    @task
    def rapid_predictions(self):
        """
        Rapid-fire predictions for stress testing
        """
        files = {'file': ('stress_test.jpg', BytesIO(self.test_image_data), 'image/jpeg')}
        self.client.post("/predict", files=files, name="STRESS /predict")


# Custom events for detailed logging
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """
    Called when the test starts
    """
    print("\n" + "=" * 80)
    print("LOAD TEST STARTED")
    print("=" * 80)
    print(f"Target host: {environment.host}")
    print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """
    Called when the test stops
    """
    print("\n" + "=" * 80)
    print("LOAD TEST COMPLETED")
    print("=" * 80)
    print(f"Test completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Print summary statistics
    stats = environment.stats.total
    print(f"\nTotal Requests: {stats.num_requests}")
    print(f"Failed Requests: {stats.num_failures}")
    print(f"Success Rate: {((stats.num_requests - stats.num_failures) / stats.num_requests * 100):.2f}%")
    print(f"Average Response Time: {stats.avg_response_time:.2f} ms")
    print(f"Min Response Time: {stats.min_response_time:.2f} ms")
    print(f"Max Response Time: {stats.max_response_time:.2f} ms")
    print(f"Requests per Second: {stats.total_rps:.2f}")

    # Percentiles
    print(f"\nResponse Time Percentiles:")
    print(f"  50th percentile: {stats.get_response_time_percentile(0.5):.2f} ms")
    print(f"  75th percentile: {stats.get_response_time_percentile(0.75):.2f} ms")
    print(f"  90th percentile: {stats.get_response_time_percentile(0.90):.2f} ms")
    print(f"  95th percentile: {stats.get_response_time_percentile(0.95):.2f} ms")
    print(f"  99th percentile: {stats.get_response_time_percentile(0.99):.2f} ms")

    print("=" * 80 + "\n")


# Usage instructions in comments
"""
HOW TO RUN LOCUST TESTS:

1. Start your Flask application:
   python app.py

2. Start Locust (web UI mode):
   locust -f locustfile.py --host=http://localhost:5000

3. Open browser and go to:
   http://localhost:8089

4. Configure test:
   - Number of users: 50-200
   - Spawn rate: 10 users/second
   - Host: http://localhost:5000
   - Click "Start Swarming"

5. Command-line mode (headless):
   locust -f locustfile.py --host=http://localhost:5000 --users 100 --spawn-rate 10 --run-time 5m --headless

6. Test different scenarios:

   # Normal load test
   locust -f locustfile.py --host=http://localhost:5000 --users 50 --spawn-rate 5 --run-time 5m

   # Stress test
   locust -f locustfile.py --host=http://localhost:5000 --users 200 --spawn-rate 20 --run-time 3m MLPredictionUser StressTestUser

   # Spike test (rapid increase)
   locust -f locustfile.py --host=http://localhost:5000 --users 500 --spawn-rate 50 --run-time 2m

DOCKER SCALING TESTS:

# Test with 1 container
docker-compose up --scale web=1
locust -f locustfile.py --host=http://localhost:5000 --users 100 --spawn-rate 10 --run-time 5m --headless

# Test with 2 containers
docker-compose up --scale web=2
locust -f locustfile.py --host=http://localhost:5000 --users 100 --spawn-rate 10 --run-time 5m --headless

# Test with 4 containers
docker-compose up --scale web=4
locust -f locustfile.py --host=http://localhost:5000 --users 100 --spawn-rate 10 --run-time 5m --headless

METRICS TO RECORD:
- Total requests
- Requests per second (RPS)
- Average latency (ms)
- P50, P95, P99 latencies
- Error rate (%)
- Number of concurrent users
- Number of Docker containers
"""