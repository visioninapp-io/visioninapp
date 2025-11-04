from graph.training.builder import builder

if __name__ == "__main__":
    user_query = input("어떤 방식으로 모델을 학습할까요?\n")
    dataset_path = "dataset@1.0.0"
    # dataset_path = "rock-paper-scissors"
    builder(user_query, dataset_path)
