from football_analytics.predict import predict_upcoming_matches


if __name__ == "__main__":
    print(predict_upcoming_matches().head(20).to_string(index=False))
