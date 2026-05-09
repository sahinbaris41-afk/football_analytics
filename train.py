import json

from football_analytics.train import train_and_save_model


if __name__ == "__main__":
    print(json.dumps(train_and_save_model(), indent=2, ensure_ascii=False))

