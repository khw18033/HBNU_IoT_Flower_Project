**🌺 Federated Learning on Raspberry Pi**

## **📖 프로젝트 개요 (Project Overview)**

본 프로젝트는 라즈베리 파이(Raspberry Pi) 기반의 IoT 엣지 환경에서 연합학습(Federated Learning) 시스템을 구축하고 그 성능과 효율성을 검증합니다.  
중앙 서버로 데이터를 전송하지 않고 로컬 디바이스에서 학습을 수행하는 Flower 프레임워크를 사용하였으며, 실제 IoT 환경의 데이터 불균형(Non-IID)과 프라이버시 보호 기법(Differential Privacy)의 영향을 실험적으로 분석하였습니다.

### **🎯 주요 목표**

1. **Feasibility**: 저사양 엣지 디바이스(Raspberry Pi 4)에서의 연합학습 구동 가능성 검증  
2. **Non-IID Robustness**: 데이터 편향 환경(0\~4 / 5\~9 분할)에서의 글로벌 모델 수렴성 확인  
3. **Resource Efficiency**: 학습 중 CPU/RAM 자원 사용량 모니터링  
4. **Privacy-Utility Trade-off**: 차분 프라이버시(DP) 적용에 따른 정확도와 보안성 간의 상충 관계 분석

## ---

**⚙️ 시스템 구성 (Architecture)**

### **🖥️ 하드웨어 (Hardware)**

* **Server (Aggregator)**: Raspberry Pi 4
* **Client 1**: Raspberry Pi 4  
* **Client 2**: Raspberry Pi 4  
* **Network**: Wi-Fi (2.4GHz)

### **🛠️ 소프트웨어 및 라이브러리 (Tech Stack)**

* **Language**: Python 3.x  
* **FL Framework**: [Flower (flwr)](https://flower.dev/)  
* **ML Engine**: TensorFlow / Keras  
* **Monitoring**: psutil (Resource logging), htop  
* **Dataset**: MNIST (Handwritten Digits)

## ---

**🚀 설치 및 실행 방법 (Getting Started)**

### **1\. 환경 설정 (Prerequisites)**

모든 기기(Server, Client 1, 2)에서 공통으로 수행합니다.

Bash

\# 시스템 업데이트  
sudo apt update && sudo apt upgrade \-y

\# 가상환경 생성 및 활성화  
python3 \-m venv fl\_project  
source fl\_project/bin/activate

\# 필수 라이브러리 설치  
pip install \--upgrade pip  
pip install "flwr\[tensorflow\]" psutil pandas numpy

### **2\. 코드 준비 및 설정**

* server.py: 서버(Aggregator) 로직. results.csv에 학습 결과를 저장합니다.  
* client1.py: 숫자 **0\~4** 데이터만 학습. client\_metrics.csv에 자원 사용량을 저장합니다.  
* client2.py: 숫자 **5\~9** 데이터만 학습.  
* **주의**: client1.py와 client2.py 하단의 server\_address를 실제 서버 IP로 수정해야 합니다.

### **3\. 실행 순서 (Running the Experiment)**

**Step 1: 서버 실행**

Bash

\# Server Terminal  
python server.py

* 서버가 시작되고 클라이언트 접속 대기 상태(Waiting for 2 clients...)가 됩니다.

**Step 2: 클라이언트 실행**

Bash

\# Client 1 Terminal  
python client1.py

\# Client 2 Terminal  
python client2.py

* 두 클라이언트가 모두 연결되면 자동으로 학습(Training)이 시작됩니다.

## ---

**📊 주요 실험 내용 및 결과 (Experiments & Results)**

### **🧪 실험 1: Non-IID 데이터 학습 (Data Distribution)**

현실적인 IoT 환경을 반영하기 위해 데이터를 의도적으로 편향되게 분배하였습니다.

* **Client 1**: 레이블 0, 1, 2, 3, 4 보유  
* **Client 2**: 레이블 5, 6, 7, 8, 9 보유  
* **결과**: 각 클라이언트는 전체 데이터의 절반만 학습했지만, 연합학습(FedAvg)을 통해 \*\*Global Accuracy 약 92%\*\*를 달성하였습니다.

### **🧪 실험 2: 자원 효율성 (Resource Monitoring)**

psutil을 사용하여 엣지 디바이스의 부하를 측정하였습니다.

* **CPU**: 로컬 학습(fit) 수행 시 100%까지 상승하지만, 통신 대기 중에는 안정화됨.  
* **RAM**: 약 40\~50% 수준에서 일정하게 유지됨 (Memory Leak 없음).  
* **결과**: Raspberry Pi 4 환경에서 발열이나 셧다운 없이 안정적인 구동이 가능함을 확인.

### **🧪 실험 3: 보안 실험 (Differential Privacy)**

가중치 전송 시 \*\*가우시안 노이즈(Gaussian Noise)\*\*를 주입하여 프라이버시를 강화하였습니다.
client1.py와 client2.py에서 noise_multiplier 값을 변경(71번째 줄)하여
Noise Multiplier를 변경할 수 있습니다.

| 실험 조건 (Condition) | Noise Multiplier | Accuracy (Round 5\) | 비고 |
| :---- | :---- | :---- | :---- |
| **Baseline** | 0.0 | **91.89%** | 기준 성능 |
| **Weak DP** | 0.1 | **92.53%** | 정규화 효과로 성능 유지 |
| **Strong DP** | 1.0 | **86.29%** | **성능 하락 (Trade-off 확인)** |

**결론**: 보안 강도를 높일수록 모델의 정확도는 하락하는 **Privacy-Utility Trade-off** 현상을 확인하였으며, 적절한 노이즈 설정이 중요함을 입증하였습니다.

## ---

**📂 파일 구조 (File Structure)**

📂 FL\_Project  
├── 📜 server.py           \# FL 서버: 모델 병합(FedAvg) 및 결과 로깅 (CSV)  
├── 📜 client1.py          \# FL 클라이언트 1: 데이터(0-4), 자원 측정, 노이즈 주입  
├── 📜 client2.py          \# FL 클라이언트 2: 데이터(5-9), 자원 측정, 노이즈 주입  
├── 📊 results.csv         \# (실행 시 생성) 라운드별 Loss, Accuracy 기록  
└── 📊 client\_metrics.csv  \# (실행 시 생성) 시간대별 CPU/RAM 사용량 기록

## ---

**👥 Contributors**

* **팀장**: 20211884 유용상
* **팀원**: 20211908 김현우, 20211880 신연수