from llm.graph.training.builder import builder
import uuid

if __name__ == "__main__":
    user_query = input("어떤 방식으로 모델을 학습할까요?\n")
    dataset_path = "dog_sample"
    # dataset_path = "rock-paper-scissors"
    job_id = str(uuid.uuid4()).replace(" ", "")
    builder(user_query, dataset_path, job_id)
