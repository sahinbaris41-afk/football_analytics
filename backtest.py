from football_analytics.backtest import run_backtest


if __name__ == "__main__":
    result = run_backtest()
    print(result["bets"].head(20).to_string(index=False))
