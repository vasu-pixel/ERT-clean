def analyze_financials(data: dict) -> dict:
    try:
        pe = data.get("trailingPE")
        peg = data.get("pegRatio")
        roe = data.get("returnOnEquity")
        margin = data.get("grossMargins")
        rev_growth = data.get("revenueGrowth")
        eps_growth = data.get("earningsGrowth")

        insights = {
            "Valuation": f"PE: {pe}, PEG: {peg}",
            "Profitability": f"ROE: {roe}, Margin: {margin}",
            "Growth": f"Revenue Growth: {rev_growth}, EPS Growth: {eps_growth}",
        }

        return insights

    except Exception as e:
        print(f"[ERROR] analyzing financials: {e}")
        return {}
    