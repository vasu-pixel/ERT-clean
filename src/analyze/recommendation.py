def generate_recommendation(data: dict) -> str:
    try:
        pe = data.get("trailingPE")
        peg = data.get("pegRatio")
        rev_growth = data.get("revenueGrowth")
        roe = data.get("returnOnEquity")

        if all(metric is not None for metric in [pe, peg, rev_growth, roe]):
            if pe < 20 and peg < 1.5 and rev_growth > 0.10 and roe > 0.15:
                return "BUY"
            elif pe < 30 and rev_growth > 0:
                return "HOLD"
            else:
                return "SELL"
        return "HOLD (insufficient data)"

    except Exception as e:
        print(f"[ERROR] generating recommendation: {e}")
        return "N/A"