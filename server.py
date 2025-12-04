import flwr as fl
import sys
import numpy as np
import csv
import os

# 결과를 저장할 파일 이름
FILENAME = "results_dp.csv"

# CSV 파일 초기화 (헤더 작성)
if not os.path.exists(FILENAME):
    with open(FILENAME, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "loss", "accuracy"])

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(self, server_round, results, failures):
        """라운드가 끝날 때마다 평가 결과를 집계하고 CSV에 저장합니다."""
        
        # 1. 기본 FedAvg 기능 수행 (Loss 집계)
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # 2. 정확도(Accuracy) 가중 평균 계산
        # results는 (client_proxy, evaluate_res) 튜플의 리스트입니다.
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        if examples:
            # 전체 데이터 개수로 나누어 가중 평균 정확도 도출
            accuracy = sum(accuracies) / sum(examples)
            print(f"[CSV Log] Round {server_round}: accuracy={accuracy}, loss={loss}")

            # 3. CSV 파일에 저장
            with open(FILENAME, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([server_round, loss, accuracy])
        else:
            accuracy = 0

        # 집계된 정확도를 Flower 시스템에 반환 (화면 출력용)
        return loss, {"accuracy": accuracy}

# 서버 전략 설정
strategy = SaveModelStrategy(
    min_available_clients=2,
    min_fit_clients=2,
    min_evaluate_clients=2,
)

if __name__ == "__main__":
    print(f"Server starting... Results will be saved to {FILENAME}")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5), # 실험을 위해 5라운드로 증가
        strategy=strategy
    )
