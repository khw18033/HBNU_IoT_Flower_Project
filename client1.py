import flwr as fl
import tensorflow as tf
import psutil
import os
import csv
import time
import numpy as np

# === [설정] 결과를 저장할 파일 이름 ===
METRICS_FILENAME = "client_metrics.csv"

# === [초기화] CSV 파일 헤더 작성 ===
if not os.path.exists(METRICS_FILENAME):
    with open(METRICS_FILENAME, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "cpu_percent", "ram_percent", "ram_used_mb"])

# 1. 데이터 로드 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# --- Client 1용 Non-IID (0~4) ---
filter_train = y_train < 5
filter_test = y_test < 5
x_train, y_train = x_train[filter_train], y_train[filter_train]
x_test, y_test = x_test[filter_test], y_test[filter_test]
# --------------------------

# 2. 모델 정의
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. 자원 측정 함수
def log_system_metrics():
    """CPU 및 RAM 사용량을 측정하여 CSV에 저장합니다."""
    cpu = psutil.cpu_percent(interval=None) # 현재 순간의 CPU 사용률
    ram = psutil.virtual_memory()
    
    with open(METRICS_FILENAME, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.time(), cpu, ram.percent, ram.used / 1024 / 1024])

# 4. Flower 클라이언트 정의
class MnistClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        
        # 학습 시작 전 자원 기록
        log_system_metrics()
        
        # 학습 수행
        model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
        
        # 학습 직후 자원 기록
        log_system_metrics()

        # === [보안 실험] Differential Privacy (노이즈 주입) ===
        # noise_multiplier: 노이즈 강도 (0.1 = 약함, 0.5 = 강함)
        # 이 값을 높일수록 보안은 강해지지만 정확도는 떨어집니다.
        noise_multiplier = 0.1 
        
        clean_weights = model.get_weights()
        noisy_weights = []
        
        for w in clean_weights:
            # 가중치의 표준편차에 비례하여 노이즈 생성
            noise_scale = np.std(w) * noise_multiplier
            noise = np.random.normal(loc=0.0, scale=noise_scale, size=w.shape)
            noisy_weights.append(w + noise)
            
        print(f"[Security] Applied DP Noise (Multiplier: {noise_multiplier})")
        # ===================================================

        # 노이즈가 섞인 가중치(noisy_weights)를 서버로 전송
        return noisy_weights, len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        return loss, len(x_test), {"accuracy": accuracy}

# 5. 클라이언트 시작
if __name__ == "__main__":
    # 서버 IP 주소 확인 필수
    fl.client.start_numpy_client(server_address="10.78.102.99:8080", client=MnistClient())
