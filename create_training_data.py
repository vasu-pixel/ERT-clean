#!/usr/bin/env python3
"""
Equity Research Training Data Generator for Ollama Finetuning

This script creates high-quality training datasets specifically designed for
finetuning Ollama models on equity research and financial analysis tasks.
"""

import os
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any
import random
import re
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EquityResearchTrainingDataGenerator:
    """Generate training data for equity research finetuning"""

    def __init__(self):
        self.output_dir = Path("training_data")
        self.output_dir.mkdir(exist_ok=True)

        # Financial terms and concepts for training
        self.financial_terms = [
            "P/E ratio", "EV/EBITDA", "ROE", "ROA", "ROIC", "FCF yield", "Debt-to-equity",
            "Current ratio", "Quick ratio", "Gross margin", "Operating margin", "Net margin",
            "Revenue growth", "Earnings growth", "EBITDA", "Enterprise value", "Market cap",
            "Book value", "Tangible book value", "Price-to-book", "PEG ratio", "Beta",
            "Dividend yield", "Payout ratio", "Working capital", "CAPEX", "Free cash flow"
        ]

        # Sector classifications
        self.sectors = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'CRM', 'ADBE'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR'],
            'Finance': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'USB'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'KMI', 'VLO'],
            'Consumer': ['WMT', 'PG', 'KO', 'PEP', 'MCD', 'NKE', 'SBUX', 'HD'],
            'Industrial': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT']
        }

    def create_instruction_response_pairs(self) -> List[Dict[str, str]]:
        """Create instruction-response pairs for supervised finetuning"""

        training_pairs = []

        # 1. Executive Summary Generation
        training_pairs.extend(self._create_executive_summary_pairs())

        # 2. Financial Analysis Pairs
        training_pairs.extend(self._create_financial_analysis_pairs())

        # 3. Valuation Analysis Pairs
        training_pairs.extend(self._create_valuation_pairs())

        # 4. Investment Thesis Pairs
        training_pairs.extend(self._create_investment_thesis_pairs())

        # 5. Risk Analysis Pairs
        training_pairs.extend(self._create_risk_analysis_pairs())

        # 6. Industry Analysis Pairs
        training_pairs.extend(self._create_industry_analysis_pairs())

        return training_pairs

    def _create_executive_summary_pairs(self) -> List[Dict[str, str]]:
        """Generate executive summary training examples"""
        pairs = []

        examples = [
            {
                "instruction": "Generate an institutional-quality executive summary for Apple Inc. (AAPL) with current price $185.50, market cap $2.9T, P/E ratio 28.5, and revenue growth 8.2%. Include investment recommendation, price target, and key highlights.",
                "response": """# EXECUTIVE SUMMARY - APPLE INC. (AAPL)

**INVESTMENT RECOMMENDATION: BUY**
**Price Target: $205.00 | Upside Potential: 10.5%**

## Investment Highlights

• **Services Growth Trajectory**: Services revenue growing at 15%+ annually with 70%+ gross margins, providing recurring revenue stability and margin expansion opportunities

• **iPhone Supercycle Catalyst**: iPhone 15 Pro adoption and upcoming AI features driving replacement cycles in premium segments with ASP expansion potential

• **Ecosystem Monetization**: Expanding installed base of 2B+ active devices creating compounding platform value and cross-selling opportunities

• **Capital Return Program**: $90B+ annual shareholder returns through dividends and buybacks, supported by strong free cash flow generation of $100B+

## Financial Overview
- **Revenue Growth**: 8.2% YoY demonstrating resilience in challenging macro environment
- **Profitability**: 28.5x P/E reflects premium valuation supported by superior margins and capital efficiency
- **Balance Sheet**: Net cash position provides strategic flexibility for M&A and investment

## Key Catalysts
- Vision Pro ecosystem development and spatial computing market expansion
- Services attach rates improvement across hardware categories
- Emerging market penetration, particularly India and Southeast Asia
- AI integration across product portfolio driving upgrade cycles

## Primary Risks
- China market exposure (20% of revenue) amid geopolitical tensions
- Regulatory scrutiny in EU and US regarding App Store policies
- Hardware replacement cycle elongation in saturated markets"""
            },
            {
                "instruction": "Create an executive summary for Microsoft Corporation (MSFT) emphasizing its cloud computing transformation, with current metrics: Price $415.25, P/E 32.1, Azure growth 29%, and Office 365 penetration expanding.",
                "response": """# EXECUTIVE SUMMARY - MICROSOFT CORPORATION (MSFT)

**INVESTMENT RECOMMENDATION: BUY**
**Price Target: $460.00 | Upside Potential: 10.8%**

## Investment Highlights

• **Azure Dominance**: 29% growth in Azure cloud services positioning Microsoft as #2 cloud provider with expanding market share against AWS, driven by enterprise hybrid solutions

• **AI Integration Leadership**: Copilot integration across Office 365 suite creating productivity multiplier effects and justifying subscription price increases, with early adoption exceeding expectations

• **Subscription Model Moat**: Office 365 and Teams creating sticky revenue streams with 90%+ retention rates and expanding seat penetration across enterprise accounts

• **Capital Allocation Excellence**: Consistent dividend growth for 20+ years combined with strategic acquisitions (LinkedIn, GitHub, Activision) demonstrating portfolio expansion capabilities

## Financial Overview
- **Revenue Mix**: 60% recurring subscription revenue providing predictable cash flows and margin expansion
- **Profitability**: 32.1x P/E supported by 35%+ operating margins and accelerating AI-driven productivity gains
- **Growth Profile**: Double-digit revenue growth across cloud, productivity, and gaming segments

## Key Catalysts
- Copilot monetization across enterprise and consumer segments driving ARPU expansion
- Azure OpenAI services capturing generative AI enterprise demand
- Gaming segment recovery post-Activision integration with content pipeline expansion
- Edge computing and IoT services addressing enterprise digital transformation needs

## Primary Risks
- Intensifying cloud competition from Google and Amazon affecting market share
- Regulatory scrutiny of AI partnerships and market concentration
- Cybersecurity threats to cloud infrastructure affecting enterprise confidence"""
            }
        ]

        for example in examples:
            pairs.append({
                "instruction": example["instruction"],
                "response": example["response"]
            })

        return pairs

    def _create_financial_analysis_pairs(self) -> List[Dict[str, str]]:
        """Generate financial analysis training examples"""
        pairs = []

        examples = [
            {
                "instruction": "Analyze the financial performance of Tesla (TSLA) focusing on profitability metrics: Gross margin 19.2%, Operating margin 8.1%, ROE 18.5%, and Free Cash Flow $7.8B. Assess quality of earnings and sustainability.",
                "response": """# FINANCIAL ANALYSIS - TESLA INC. (TSLA)

## Profitability Assessment

**Gross Margin Analysis (19.2%)**
• Automotive gross margins improving due to manufacturing scale efficiencies and localization strategies
• Services gross margins (energy, supercharging) contributing incremental profitability
• Regulatory credit revenue (5% of automotive revenue) provides margin cushion but declining over time
• Manufacturing learning curve demonstrating consistent cost reduction of 15-20% per doubling of production

**Operating Leverage (8.1% Operating Margin)**
• Operating leverage inflection achieved with fixed cost absorption across higher volume base
• R&D intensity (3.1% of revenue) supporting autonomous driving and energy storage innovation
• SG&A efficiency improving as brand strength reduces marketing requirements
• Quality control investments addressing service capacity constraints

## Capital Efficiency Metrics

**Return on Equity (18.5%)**
• Superior ROE reflecting asset-light manufacturing approach and vertical integration benefits
• Working capital optimization through direct-to-consumer model eliminating dealer inventory
• Gigafactory utilization rates approaching 80-90% efficiency targets

**Free Cash Flow Generation ($7.8B)**
• Positive operating cash flow driven by strong unit economics and working capital management
• CAPEX intensity moderating as factory construction phases complete
• Cash conversion cycle improvements through supplier payment optimization

## Quality of Earnings Assessment

**Sustainability Factors**
• Recurring revenue streams developing through software updates and autonomous features
• Energy business contributing stable, utility-like cash flows
• Supercharger network creating ecosystem moat and third-party revenue opportunities

**Risk Factors**
• Regulatory credit dependence creating earnings volatility
• Commodity price exposure affecting battery input costs
• Competition intensification from traditional OEMs and Chinese manufacturers"""
            }
        ]

        for example in examples:
            pairs.append({
                "instruction": example["instruction"],
                "response": example["response"]
            })

        return pairs

    def _create_valuation_pairs(self) -> List[Dict[str, str]]:
        """Generate valuation analysis training examples"""
        pairs = []

        examples = [
            {
                "instruction": "Conduct a comprehensive DCF valuation for Amazon (AMZN) with the following assumptions: AWS revenue growth 20-25%, E-commerce growth 8-12%, Operating margin expansion to 12% by 2028, Terminal growth rate 3%, and WACC 8.5%.",
                "response": """# DCF VALUATION ANALYSIS - AMAZON.COM INC. (AMZN)

## Revenue Projections (5-Year Forecast)

**AWS Segment**
- 2024E: $105B (22% growth) - Maintaining market leadership in cloud infrastructure
- 2028E: $225B (20% CAGR) - Expanding enterprise adoption and AI/ML services integration
- Key drivers: Digital transformation acceleration, hybrid cloud adoption, generative AI demand

**E-commerce Segment**
- 2024E: $390B (10% growth) - Benefiting from Prime membership expansion and international growth
- 2028E: $560B (9% CAGR) - Mature market growth with third-party marketplace acceleration
- Key drivers: Advertising services integration, logistics network monetization, emerging market penetration

## Margin Expansion Analysis

**Operating Margin Progression**
- Current: 5.7% (depressed by investment cycle and macro headwinds)
- Target: 12% by 2028 (AWS margin stabilization + e-commerce efficiency gains)
- AWS margins: 35% (sustainable due to scale economics and switching costs)
- E-commerce margins: 6-8% (fulfillment automation and advertising revenue mix)

## DCF Model Inputs

**Cost of Capital (WACC: 8.5%)**
- Risk-free rate: 4.5%
- Equity risk premium: 6.0%
- Beta: 1.15 (cyclical consumer/technology hybrid)
- Tax rate: 25% (normalized effective rate)

**Terminal Value Assumptions**
- Terminal growth rate: 3.0% (reflecting long-term GDP growth)
- Terminal FCF margin: 15% (mature technology company comparable)
- Perpetuity multiple: 20x FCF (premium for platform economics)

## Valuation Output

**DCF Calculation**
- Sum of PV of FCF (2024-2028): $485B
- Terminal value: $1,150B
- Present value of terminal: $780B
- Enterprise value: $1,265B
- Less: Net debt ($15B)
- Equity value: $1,280B

**Price Target: $125 per share**
- Current price: $105
- Upside potential: 19%
- Sensitivity analysis range: $115-$135

## Comparable Company Analysis

**EV/Revenue Multiples**
- AWS segment: 8.5x (vs. cloud peers 6-12x)
- E-commerce: 1.8x (vs. retail peers 1.2-2.5x)
- Blended multiple: 3.2x 2025E revenue

**Risk-Adjusted Scenarios**
- Bull case ($140): Faster AWS growth + margin expansion
- Base case ($125): Model assumptions achieved
- Bear case ($95): Competitive pressure + macro headwinds"""
            }
        ]

        for example in examples:
            pairs.append({
                "instruction": example["instruction"],
                "response": example["response"]
            })

        return pairs

    def _create_investment_thesis_pairs(self) -> List[Dict[str, str]]:
        """Generate investment thesis training examples"""
        pairs = []

        examples = [
            {
                "instruction": "Develop a balanced investment thesis for NVIDIA (NVDA) considering AI/ML demand drivers, competitive positioning, and valuation concerns. Include both bull and bear case scenarios with specific catalysts and risks.",
                "response": """# INVESTMENT THESIS - NVIDIA CORPORATION (NVDA)

## Bull Case: AI Revolution Beneficiary

**Thesis: NVIDIA is uniquely positioned to capitalize on the generative AI infrastructure buildout, with sustainable competitive advantages in high-performance computing.**

### Key Catalysts (12-18 months)
• **Data Center Demand Surge**: Enterprise AI adoption driving 50%+ growth in H100/A100 demand with pricing power maintained
• **Software Ecosystem Expansion**: CUDA platform lock-in effects strengthening as AI developers build on NVIDIA architecture
• **Automotive AI Integration**: Autonomous vehicle partnerships with Mercedes, BMW accelerating Drive platform adoption
• **Edge AI Deployment**: Omniverse and edge computing solutions addressing industrial metaverse applications

### Competitive Moat Analysis
- **Technology Leadership**: 5-7 year lead in AI chip architecture with established CUDA ecosystem
- **Manufacturing Partnerships**: Exclusive TSMC advanced node allocation securing supply chain advantages
- **Customer Stickiness**: Enterprise AI workloads trained on NVIDIA infrastructure creating switching costs
- **R&D Investment**: 25% of revenue reinvested in next-generation architecture development

### Financial Upside
- **Revenue Potential**: $150B+ by 2026 (vs. $60B current) driven by AI infrastructure spending
- **Margin Expansion**: Data center gross margins sustaining 75%+ due to specialized chip premium
- **Free Cash Flow**: $80B+ annual generation enabling aggressive R&D and shareholder returns

## Bear Case: Valuation and Competition Risks

**Thesis: Current valuation assumes perfect execution while underestimating competitive threats and cyclical demand patterns.**

### Key Risk Factors
• **Competition Intensification**: AMD MI300X, Intel Gaudi, and custom AI chips from hyperscalers reducing market share
• **Geopolitical Exposure**: China restrictions limiting 25% of addressable market with compliance costs increasing
• **Customer Concentration**: Hyperscaler dependency (70% of data center revenue) creating negotiating power imbalance
• **Cyclical Downturn**: Historical semiconductor cycle patterns suggesting demand normalization in 2025-2026

### Valuation Concerns
- **Multiple Expansion**: 65x P/E trading at 2x premium to historical technology averages
- **Growth Sustainability**: 100%+ revenue growth rates historically unsustainable beyond 2-3 years
- **Competition Impact**: Market share erosion could compress margins to 45-50% normalized levels

### Downside Scenarios
- **Market Share Loss**: AMD/Intel gaining 20-30% data center share by 2026
- **Demand Normalization**: AI spending growth slowing to 15-20% as infrastructure buildout matures
- **Regulatory Risks**: Export controls expanding to additional geographies affecting revenue base

## Balanced Investment View

**Recommendation: HOLD with Price Target $485**
- **Current Price**: $520
- **Upside Scenario**: $650 (AI dominance maintained)
- **Downside Scenario**: $350 (competition + cycle impact)

### Risk-Adjusted Position Sizing
- **Growth Portfolios**: 3-5% allocation capturing AI theme exposure
- **Value Portfolios**: Underweight until valuation compression
- **Timing Strategy**: Dollar-cost averaging approach given volatility expectations

### Monitoring Metrics
- Data center revenue growth vs. hyperscaler CAPEX guidance
- CUDA market share in AI training/inference workloads
- Competitive product launches and customer wins
- Geopolitical developments affecting export restrictions"""
            }
        ]

        for example in examples:
            pairs.append({
                "instruction": example["instruction"],
                "response": example["response"]
            })

        return pairs

    def _create_risk_analysis_pairs(self) -> List[Dict[str, str]]:
        """Generate risk analysis training examples"""
        pairs = []

        examples = [
            {
                "instruction": "Conduct comprehensive risk analysis for Johnson & Johnson (JNJ) covering business risks, regulatory exposure, litigation concerns, and ESG factors. Quantify potential financial impact where possible.",
                "response": """# COMPREHENSIVE RISK ANALYSIS - JOHNSON & JOHNSON (JNJ)

## Business & Operational Risks

**Pharmaceutical Pipeline Risk (HIGH)**
- **Patent Cliff Exposure**: $15B+ in revenue at risk from 2025-2028 patent expirations (Stelara, Imbruvica)
- **R&D Success Rate**: 12% clinical trial success rate below industry average of 15-18%
- **Biosimilar Competition**: Remicade biosimilars reducing revenue by $2B annually since 2018
- **Impact Assessment**: 8-12% revenue headwind over 3-year period without pipeline replacement

**Supply Chain Vulnerabilities (MEDIUM)**
- **Geographic Concentration**: 40% of API manufacturing in India/China creating geopolitical risk
- **Quality Control**: FDA inspections identifying deficiencies at 3 key facilities in 2023
- **Raw Material Inflation**: 5-8% annual cost increases for specialized pharmaceutical inputs
- **Mitigation**: $2B supply chain diversification program reducing concentration by 2026

## Regulatory & Compliance Risks

**FDA Oversight Intensification (MEDIUM-HIGH)**
- **Device Division Scrutiny**: Medical device recalls increasing regulatory burden and approval timelines
- **Clinical Trial Standards**: Enhanced post-market surveillance requirements increasing compliance costs by 15%
- **International Harmonization**: EU MDR compliance requiring additional clinical evidence for device approvals

**Pricing Pressure (HIGH)**
- **Medicare Negotiation**: IRA drug pricing negotiations potentially affecting 8-10 key products
- **International Reference Pricing**: European pricing controls limiting premium pricing strategies
- **Financial Impact**: $3-5B potential revenue impact by 2028 from pricing constraints

## Litigation & Legal Exposure

**Talc Litigation Legacy (HIGH)**
- **Outstanding Claims**: 38,000+ pending lawsuits with settlement negotiations ongoing
- **Financial Reserve**: $9.9B reserved but potential exposure estimated at $15-20B
- **Bankruptcy Strategy**: Subsidiary bankruptcy filing creating legal uncertainty
- **Timeline**: Resolution expected by 2025-2026 with final settlement framework

**Opioid Litigation (MEDIUM)**
- **State Settlements**: $5B commitment over 9 years for opioid crisis resolution
- **Ongoing Investigations**: DOJ criminal investigation creating additional exposure risk
- **Reputation Impact**: Brand damage affecting physician prescribing patterns

## ESG & Sustainability Risks

**Environmental Liability (MEDIUM)**
- **Manufacturing Footprint**: 260 facilities globally with varying environmental standards
- **Water Usage**: Pharmaceutical manufacturing requiring significant water resources in water-stressed regions
- **Waste Management**: Hazardous waste disposal creating long-term environmental liability

**Social License Risks (MEDIUM)**
- **Access to Medicine**: Pricing criticism in developing markets affecting government relations
- **Clinical Trial Ethics**: Ensuring informed consent and equitable patient access across global studies
- **Supply Chain Labor**: Third-party supplier compliance with labor standards requiring enhanced monitoring

## Financial Risk Quantification

**Earnings-at-Risk Analysis**
- **Base Case Impact**: -5% to -8% EPS from combined risk factors over 3-year period
- **Stress Scenario**: -15% to -20% EPS if multiple high-impact risks materialize simultaneously
- **Risk-Adjusted Valuation**: DCF discount rate increased by 50-75 bps for heightened uncertainty

**Liquidity & Credit Profile**
- **Debt Maturity**: $28B debt maturities through 2026 requiring refinancing in higher rate environment
- **Credit Rating**: AA- stable outlook but potential downgrade if litigation settlements exceed $20B
- **Interest Coverage**: 12x EBITDA/interest providing substantial cushion for adverse scenarios

## Risk Mitigation Strategies

**Portfolio Diversification**
- **Innovation Investment**: $15B annual R&D focusing on oncology, immunology, neuroscience
- **Geographic Expansion**: Emerging market penetration reducing developed market dependence
- **Platform Technologies**: mRNA, CAR-T capabilities building next-generation treatment options

**Financial Hedging**
- **Currency Exposure**: 50% of revenue hedged against EUR, JPY fluctuations
- **Interest Rate Risk**: Fixed-rate debt structure minimizing refinancing risk
- **Insurance Coverage**: $2B product liability and D&O coverage for litigation protection"""
            }
        ]

        for example in examples:
            pairs.append({
                "instruction": example["instruction"],
                "response": example["response"]
            })

        return pairs

    def _create_industry_analysis_pairs(self) -> List[Dict[str, str]]:
        """Generate industry analysis training examples"""
        pairs = []

        examples = [
            {
                "instruction": "Analyze the electric vehicle industry dynamics, competitive landscape, and growth outlook. Focus on market size, key players, technology trends, and regulatory drivers affecting the sector.",
                "response": """# ELECTRIC VEHICLE INDUSTRY ANALYSIS

## Market Size & Growth Dynamics

**Global EV Market Trajectory**
- **Current Market**: 14.1 million EVs sold globally in 2023 (18% of total auto sales)
- **Growth Projection**: 35-40% CAGR through 2030 reaching 70+ million annual sales
- **Market Value**: $500B current market expanding to $1.7T by 2030
- **Regional Leadership**: China (60% market share), Europe (25%), North America (10%)

**Adoption Curve Analysis**
- **Technology Adoption**: Crossing 15% penetration threshold indicating mainstream adoption phase
- **Price Parity**: Battery EVs reaching cost parity with ICE vehicles by 2025-2027
- **Infrastructure Scaling**: 2.7 million public charging points globally, targeting 15 million by 2030

## Competitive Landscape Assessment

**Market Leadership Tiers**

**Tier 1: Global Scale Leaders**
- **Tesla (TSLA)**: 20% global market share, software/charging ecosystem advantages
- **BYD (China)**: 17% market share, vertical integration in batteries and semiconductors
- **Volkswagen Group**: 8% market share, platform standardization across brands

**Tier 2: Regional Champions**
- **CATL/LG Energy**: Battery technology leaders controlling 65% of global cell capacity
- **NIO/XPeng (China)**: Premium positioning with battery-as-a-service innovation
- **GM/Ford (US)**: Legacy OEM transformation with $50B+ EV investment commitments

**Tier 3: Emerging Disruptors**
- **Rivian/Lucid**: Specialized segments (trucks/luxury) with technology differentiation
- **European OEMs**: Mercedes EQS, BMW iX targeting premium segments

## Technology Evolution Trends

**Battery Technology Roadmap**
- **Energy Density**: Lithium-ion advancing from 250 Wh/kg to 400+ Wh/kg by 2030
- **Cost Reduction**: Battery pack costs declining from $132/kWh (2023) to $80/kWh (2030)
- **Chemistry Innovation**: LFP batteries gaining share (40%) due to cost/safety advantages
- **Solid State**: Commercial deployment expected 2027-2030 for premium applications

**Charging Infrastructure**
- **Fast Charging**: 350kW+ charging enabling 10-80% charge in 15-20 minutes
- **Network Effects**: Tesla Supercharger opening to other brands expanding interoperability
- **Grid Integration**: Vehicle-to-grid technology creating energy storage value proposition

**Autonomous Integration**
- **Software-Defined Vehicles**: OTA updates and subscription services creating recurring revenue
- **L3/L4 Autonomy**: Highway pilot features standard by 2027-2030
- **Data Monetization**: Connected vehicle data enabling insurance, maintenance, mobility services

## Regulatory & Policy Drivers

**Emission Standards Tightening**
- **EU**: 100% zero-emission new car sales mandate by 2035
- **California/CARB States**: 68% ZEV sales requirement by 2030
- **China**: NEV mandate requiring 40% alternative fuel vehicles by 2030

**Incentive Structures**
- **US IRA**: $7,500 federal tax credit with domestic content requirements
- **EU Green Deal**: €3B in EV charging infrastructure investment
- **China**: Purchase tax exemption and license plate advantages in major cities

**Trade & Supply Chain Policy**
- **US-China Tensions**: Tariffs and restrictions affecting battery supply chains
- **Critical Minerals**: Strategic stockpiling of lithium, cobalt, nickel, rare earths
- **Domestic Content**: Local assembly and battery manufacturing requirements

## Investment Outlook & Risks

**Capital Requirements**
- **Industry CAPEX**: $1.2T required investment through 2030 for production capacity
- **Battery Manufacturing**: $200B investment needed for 4 TWh global capacity
- **Charging Infrastructure**: $90B required for adequate charging network density

**Key Risk Factors**
- **Raw Material Inflation**: Lithium prices volatility affecting battery costs
- **Geopolitical Supply Chain**: Concentration in China for battery materials and processing
- **Grid Capacity**: Electrical infrastructure upgrades required for mass adoption
- **Technology Obsolescence**: Rapid innovation cycles creating stranded asset risk

## Sector Recommendations

**Investment Strategy**
- **Overweight**: Battery technology leaders (CATL, TSLA) and charging infrastructure
- **Selective**: Traditional OEMs with credible EV transformation (GM, F, STLA)
- **Underweight**: Pure-play EV startups with limited scale and funding constraints

**Timing Considerations**
- **Early Cycle**: Infrastructure and technology enablers (charging, batteries, semiconductors)
- **Mid Cycle**: Established OEMs with platform scalability and brand strength
- **Late Cycle**: Autonomous and software monetization plays requiring mass adoption"""
            }
        ]

        for example in examples:
            pairs.append({
                "instruction": example["instruction"],
                "response": example["response"]
            })

        return pairs

    def fetch_real_market_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch real market data for training examples"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            financials = stock.financials

            return {
                'ticker': ticker,
                'price': info.get('currentPrice', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return {}

    def generate_dynamic_training_examples(self, num_examples: int = 100) -> List[Dict[str, str]]:
        """Generate training examples using real market data"""
        training_pairs = []

        # Get random tickers from different sectors
        all_tickers = []
        for sector_tickers in self.sectors.values():
            all_tickers.extend(sector_tickers)

        selected_tickers = random.sample(all_tickers, min(num_examples // 4, len(all_tickers)))

        for ticker in selected_tickers:
            data = self.fetch_real_market_data(ticker)
            if data:
                # Generate different types of analysis for each ticker
                training_pairs.extend(self._create_dynamic_examples(data))

        return training_pairs

    def _create_dynamic_examples(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create training examples based on real market data"""
        pairs = []

        ticker = data['ticker']
        price = data.get('price', 100)
        pe_ratio = data.get('pe_ratio', 20)
        sector = data.get('sector', 'Technology')

        # Executive Summary
        pairs.append({
            "instruction": f"Generate an executive summary for {ticker} with current price ${price:.2f}, P/E ratio {pe_ratio:.1f}, in the {sector} sector.",
            "response": f"# EXECUTIVE SUMMARY - {ticker}\n\n**Current Price**: ${price:.2f}\n**P/E Ratio**: {pe_ratio:.1f}\n**Sector**: {sector}\n\n[Analysis would continue with specific insights based on current market conditions and company fundamentals...]"
        })

        return pairs

    def save_training_dataset(self, training_pairs: List[Dict[str, str]], filename: str = "equity_research_training.jsonl"):
        """Save training data in JSONL format for Ollama finetuning"""

        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in training_pairs:
                json_line = json.dumps({
                    "instruction": pair["instruction"],
                    "response": pair["response"]
                }, ensure_ascii=False)
                f.write(json_line + '\n')

        logger.info(f"Training dataset saved to {output_path}")
        logger.info(f"Total training examples: {len(training_pairs)}")

        return output_path

    def create_validation_set(self, training_pairs: List[Dict[str, str]], validation_split: float = 0.15):
        """Split data into training and validation sets"""

        random.shuffle(training_pairs)
        split_idx = int(len(training_pairs) * (1 - validation_split))

        train_data = training_pairs[:split_idx]
        val_data = training_pairs[split_idx:]

        # Save training set
        train_path = self.save_training_dataset(train_data, "equity_research_train.jsonl")

        # Save validation set
        val_path = self.save_training_dataset(val_data, "equity_research_validation.jsonl")

        logger.info(f"Training examples: {len(train_data)}")
        logger.info(f"Validation examples: {len(val_data)}")

        return train_path, val_path

    def generate_complete_dataset(self):
        """Generate complete training dataset for equity research finetuning"""

        logger.info("Starting training dataset generation...")

        # 1. Create instruction-response pairs
        logger.info("Creating instruction-response pairs...")
        static_pairs = self.create_instruction_response_pairs()

        # 2. Generate dynamic examples with real data
        logger.info("Generating dynamic examples with real market data...")
        dynamic_pairs = self.generate_dynamic_training_examples(50)

        # 3. Combine all training data
        all_pairs = static_pairs + dynamic_pairs
        logger.info(f"Total training examples generated: {len(all_pairs)}")

        # 4. Create train/validation split
        train_path, val_path = self.create_validation_set(all_pairs)

        # 5. Generate metadata
        metadata = {
            "dataset_info": {
                "name": "Equity Research Training Dataset",
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "total_examples": len(all_pairs),
                "training_examples": len(all_pairs) - int(len(all_pairs) * 0.15),
                "validation_examples": int(len(all_pairs) * 0.15)
            },
            "content_categories": {
                "executive_summaries": "Investment recommendations and key highlights",
                "financial_analysis": "Profitability, efficiency, and quality metrics",
                "valuation_analysis": "DCF models and comparable company analysis",
                "investment_thesis": "Bull/bear cases and catalysts",
                "risk_analysis": "Business, regulatory, and financial risks",
                "industry_analysis": "Sector dynamics and competitive landscape"
            },
            "model_recommendations": {
                "base_model": "llama3.1:8b",
                "finetuning_method": "LoRA",
                "recommended_epochs": 3,
                "learning_rate": 2e-5,
                "batch_size": 4
            }
        }

        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Dataset generation complete!")
        logger.info(f"Training data: {train_path}")
        logger.info(f"Validation data: {val_path}")
        logger.info(f"Metadata: {metadata_path}")

        return {
            "train_path": train_path,
            "validation_path": val_path,
            "metadata_path": metadata_path,
            "total_examples": len(all_pairs)
        }

def main():
    """Main function to generate training dataset"""
    generator = EquityResearchTrainingDataGenerator()
    result = generator.generate_complete_dataset()

    print("\n🎯 Training Dataset Generation Complete!")
    print(f"📄 Training examples: {result['total_examples']}")
    print(f"📁 Files saved to: training_data/")
    print(f"🚀 Ready for Ollama finetuning!")

if __name__ == "__main__":
    main()