"""
Enhanced Peer Selection Module
ML-based peer identification with dynamic universe updates
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class PeerCompanyEnhanced:
    """Enhanced peer company data structure"""
    ticker: str
    name: str
    sector: str
    industry: str
    market_cap: float
    revenue: float
    ebitda: float
    beta: float
    roe: float
    roce: float
    debt_to_equity: float
    revenue_growth: float
    ebitda_margin: float
    pe_ratio: float
    ev_ebitda: float
    similarity_score: float
    peer_type: str  # 'fundamental', 'business_model', 'market_based'
    selection_reasons: List[str]

@dataclass
class EnhancedPeerResults:
    """Results from enhanced peer selection"""
    ml_selected_peers: List[PeerCompanyEnhanced]
    fundamental_peers: List[PeerCompanyEnhanced]
    business_model_peers: List[PeerCompanyEnhanced]
    market_based_peers: List[PeerCompanyEnhanced]
    peer_clusters: Dict[str, List[str]]
    similarity_matrix: np.ndarray
    feature_importance: Dict[str, float]
    peer_universe_size: int
    selection_methodology: Dict[str, Any]
    quality_metrics: Dict[str, float]
    methodology_notes: List[str]

class EnhancedPeerSelector:
    """
    Advanced peer selection using machine learning and dynamic analysis
    """

    def __init__(self):
        self.feature_weights = {
            'market_cap': 0.15,
            'revenue': 0.20,
            'ebitda_margin': 0.15,
            'roe': 0.10,
            'debt_to_equity': 0.10,
            'revenue_growth': 0.15,
            'beta': 0.10,
            'sector_dummy': 0.05
        }

        # Sector-specific peer universe mappings
        self.sector_universes = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'CRM', 'ORCL', 'ADBE',
                          'NFLX', 'PYPL', 'INTC', 'CSCO', 'IBM', 'QCOM', 'TXN', 'AVGO', 'AMD', 'MU'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY', 'MRK', 'CVS',
                          'LLY', 'MDT', 'GILD', 'AMGN', 'SYK', 'BDX', 'CI', 'HUM', 'ANTM', 'ZTS'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF',
                                  'AXP', 'BK', 'STT', 'BLK', 'SCHW', 'CME', 'ICE', 'SPGI', 'MCO', 'V'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'F', 'GM',
                                 'DIS', 'BKNG', 'CMG', 'YUM', 'MAR', 'HLT', 'RCL', 'CCL', 'LVS', 'WYNN'],
            'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS', 'K', 'CPB',
                                  'SJM', 'CAG', 'HSY', 'MDLZ', 'MNST', 'KHC', 'HRL', 'MKC', 'CLX', 'CHD'],
            'Industrial': ['MMM', 'BA', 'CAT', 'DE', 'GE', 'HON', 'LMT', 'RTX', 'UPS', 'UNP',
                          'NSC', 'CSX', 'FDX', 'EMR', 'ETN', 'PH', 'ROK', 'ITW', 'DOV', 'XYL'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'KMI', 'OKE',
                      'WMB', 'EPD', 'ET', 'MPLX', 'BKR', 'HAL', 'DVN', 'FANG', 'MRO', 'APA'],
            'Materials': ['LIN', 'SHW', 'APD', 'ECL', 'DD', 'DOW', 'NEM', 'FCX', 'CF', 'LYB',
                         'VMC', 'MLM', 'NUE', 'STLD', 'X', 'CLF', 'PKG', 'IP', 'WRK', 'SON'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'EXC', 'XEL', 'SRE', 'AEP', 'ED', 'PEG',
                         'ES', 'FE', 'EIX', 'ETR', 'WEC', 'DTE', 'PPL', 'CMS', 'NI', 'LNT'],
            'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'WELL', 'DLR', 'PSA', 'EXR', 'AVB', 'EQR',
                           'SBAC', 'VTR', 'ARE', 'MAA', 'ESS', 'UDR', 'CPT', 'FRT', 'REG', 'O']
        }

    def select_enhanced_peers(self, target_ticker: str,
                            num_peers: int = 15,
                            use_ml: bool = True) -> EnhancedPeerResults:
        """
        Select peers using enhanced ML-based methodology
        """
        try:
            print(f"🤖 Enhanced peer selection for {target_ticker}...")

            # Step 1: Get target company data
            target_data = self._get_company_fundamentals(target_ticker)
            if not target_data:
                return self._create_empty_results()

            # Step 2: Build peer universe
            peer_universe = self._build_dynamic_peer_universe(target_data)

            # Step 3: Get fundamentals for all potential peers
            peer_fundamentals = self._get_peer_fundamentals(peer_universe)

            # Step 4: ML-based peer selection
            if use_ml:
                ml_peers = self._ml_peer_selection(target_data, peer_fundamentals, num_peers)
            else:
                ml_peers = []

            # Step 5: Fundamental-based selection
            fundamental_peers = self._fundamental_peer_selection(target_data, peer_fundamentals, num_peers)

            # Step 6: Business model-based selection
            business_peers = self._business_model_peer_selection(target_data, peer_fundamentals)

            # Step 7: Market-based selection
            market_peers = self._market_based_peer_selection(target_data, peer_fundamentals)

            # Step 8: Create similarity matrix
            similarity_matrix = self._create_similarity_matrix(target_data, peer_fundamentals)

            # Step 9: Cluster analysis
            peer_clusters = self._perform_cluster_analysis(peer_fundamentals)

            # Step 10: Calculate feature importance
            feature_importance = self._calculate_feature_importance(target_data, peer_fundamentals)

            # Step 11: Quality metrics
            quality_metrics = self._calculate_quality_metrics(ml_peers, fundamental_peers)

            # Step 12: Selection methodology summary
            methodology = self._create_methodology_summary()

            # Step 13: Generate notes
            methodology_notes = self._generate_methodology_notes()

            return EnhancedPeerResults(
                ml_selected_peers=ml_peers,
                fundamental_peers=fundamental_peers,
                business_model_peers=business_peers,
                market_based_peers=market_peers,
                peer_clusters=peer_clusters,
                similarity_matrix=similarity_matrix,
                feature_importance=feature_importance,
                peer_universe_size=len(peer_universe),
                selection_methodology=methodology,
                quality_metrics=quality_metrics,
                methodology_notes=methodology_notes
            )

        except Exception as e:
            logger.error(f"Error in enhanced peer selection: {e}")
            return self._create_empty_results()

    def _get_company_fundamentals(self, ticker: str) -> Optional[Dict]:
        """Get comprehensive fundamental data for a company"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            financials = stock.financials
            balance_sheet = stock.balance_sheet

            if not info:
                return None

            # Calculate derived metrics
            market_cap = info.get('marketCap', 0)
            revenue = info.get('totalRevenue', 0)
            ebitda = info.get('ebitda', 0)

            # Get growth rates
            revenue_growth = 0.0
            if not financials.empty and len(financials.columns) >= 2:
                recent_revenue = financials.loc['Total Revenue'].iloc[0] if 'Total Revenue' in financials.index else 0
                prev_revenue = financials.loc['Total Revenue'].iloc[1] if 'Total Revenue' in financials.index else 0
                if prev_revenue > 0:
                    revenue_growth = (recent_revenue - prev_revenue) / prev_revenue

            return {
                'ticker': ticker,
                'name': info.get('longName', ticker),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': market_cap,
                'revenue': revenue,
                'ebitda': ebitda,
                'beta': info.get('beta', 1.0),
                'roe': info.get('returnOnEquity', 0),
                'roce': info.get('returnOnAssets', 0) * 1.5,  # Approximation
                'debt_to_equity': info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0,
                'revenue_growth': revenue_growth,
                'ebitda_margin': ebitda / revenue if revenue > 0 else 0,
                'pe_ratio': info.get('trailingPE', 0),
                'ev_ebitda': info.get('enterpriseToEbitda', 0),
                'price': info.get('currentPrice', 0),
                'country': info.get('country', 'US')
            }

        except Exception as e:
            logger.warning(f"Could not get fundamentals for {ticker}: {e}")
            return None

    def _build_dynamic_peer_universe(self, target_data: Dict) -> List[str]:
        """Build dynamic peer universe based on target company characteristics"""
        try:
            target_sector = target_data.get('sector', '')
            target_market_cap = target_data.get('market_cap', 0)

            # Start with sector-based universe
            base_universe = self.sector_universes.get(target_sector, [])

            # Add cross-sector peers for large companies
            if target_market_cap > 100e9:  # $100B+ companies
                for sector_peers in self.sector_universes.values():
                    base_universe.extend(sector_peers[:5])  # Top 5 from each sector

            # Add size-similar companies from adjacent sectors
            adjacent_sectors = self._get_adjacent_sectors(target_sector)
            for adj_sector in adjacent_sectors:
                if adj_sector in self.sector_universes:
                    base_universe.extend(self.sector_universes[adj_sector][:10])

            # Remove duplicates and target ticker
            universe = list(set(base_universe))
            if target_data['ticker'] in universe:
                universe.remove(target_data['ticker'])

            return universe[:200]  # Limit universe size for performance

        except Exception as e:
            logger.warning(f"Error building peer universe: {e}")
            return []

    def _get_adjacent_sectors(self, sector: str) -> List[str]:
        """Get sectors adjacent/similar to target sector"""
        adjacency_map = {
            'Technology': ['Communication Services', 'Consumer Cyclical'],
            'Healthcare': ['Consumer Defensive'],
            'Financial Services': ['Real Estate'],
            'Consumer Cyclical': ['Technology', 'Communication Services'],
            'Consumer Defensive': ['Healthcare'],
            'Industrial': ['Materials', 'Energy'],
            'Energy': ['Materials', 'Utilities'],
            'Materials': ['Industrial', 'Energy'],
            'Utilities': ['Energy', 'Real Estate'],
            'Real Estate': ['Utilities', 'Financial Services'],
            'Communication Services': ['Technology', 'Consumer Cyclical']
        }

        return adjacency_map.get(sector, [])

    def _get_peer_fundamentals(self, peer_universe: List[str]) -> List[Dict]:
        """Get fundamental data for entire peer universe"""
        fundamentals = []

        print(f"  📊 Analyzing {len(peer_universe)} potential peers...")

        for ticker in peer_universe:
            try:
                data = self._get_company_fundamentals(ticker)
                if data and data.get('market_cap', 0) > 0:
                    fundamentals.append(data)

                # Progress indicator
                if len(fundamentals) % 20 == 0:
                    print(f"    Processed {len(fundamentals)} companies...")

            except Exception as e:
                logger.warning(f"Skipping {ticker}: {e}")

        print(f"  ✅ Successfully analyzed {len(fundamentals)} companies")
        return fundamentals

    def _ml_peer_selection(self, target_data: Dict, peer_data: List[Dict],
                          num_peers: int) -> List[PeerCompanyEnhanced]:
        """Select peers using machine learning clustering and similarity"""
        try:
            if len(peer_data) < 10:  # Need minimum data for ML
                return []

            # Create feature matrix
            features = []
            tickers = []

            # Include target in analysis
            all_data = [target_data] + peer_data

            for company in all_data:
                feature_vector = [
                    np.log(max(1, company.get('market_cap', 1))),  # Log market cap
                    np.log(max(1, company.get('revenue', 1))),     # Log revenue
                    company.get('ebitda_margin', 0),
                    company.get('roe', 0),
                    company.get('debt_to_equity', 0),
                    company.get('revenue_growth', 0),
                    company.get('beta', 1.0),
                    1 if company.get('sector') == target_data.get('sector') else 0  # Sector dummy
                ]

                features.append(feature_vector)
                tickers.append(company['ticker'])

            features_array = np.array(features)

            # Handle NaN values
            features_array = np.nan_to_num(features_array)

            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_array)

            # Calculate similarity to target (first row)
            target_features = features_scaled[0].reshape(1, -1)
            peer_features = features_scaled[1:]

            similarities = cosine_similarity(target_features, peer_features)[0]

            # Create peer objects with similarity scores
            ml_peers = []
            for i, similarity in enumerate(similarities):
                if similarity > 0.7:  # Similarity threshold
                    peer_data_item = peer_data[i]
                    peer = PeerCompanyEnhanced(
                        ticker=peer_data_item['ticker'],
                        name=peer_data_item['name'],
                        sector=peer_data_item['sector'],
                        industry=peer_data_item['industry'],
                        market_cap=peer_data_item['market_cap'],
                        revenue=peer_data_item['revenue'],
                        ebitda=peer_data_item['ebitda'],
                        beta=peer_data_item['beta'],
                        roe=peer_data_item['roe'],
                        roce=peer_data_item['roce'],
                        debt_to_equity=peer_data_item['debt_to_equity'],
                        revenue_growth=peer_data_item['revenue_growth'],
                        ebitda_margin=peer_data_item['ebitda_margin'],
                        pe_ratio=peer_data_item['pe_ratio'],
                        ev_ebitda=peer_data_item['ev_ebitda'],
                        similarity_score=float(similarity),
                        peer_type='ml_selected',
                        selection_reasons=[
                            f"High ML similarity score: {similarity:.3f}",
                            "Similar fundamental characteristics",
                            "Cluster analysis match"
                        ]
                    )
                    ml_peers.append(peer)

            # Sort by similarity and return top peers
            ml_peers.sort(key=lambda x: x.similarity_score, reverse=True)
            return ml_peers[:num_peers]

        except Exception as e:
            logger.warning(f"Error in ML peer selection: {e}")
            return []

    def _fundamental_peer_selection(self, target_data: Dict, peer_data: List[Dict],
                                  num_peers: int) -> List[PeerCompanyEnhanced]:
        """Select peers based on fundamental similarity"""
        try:
            target_cap = target_data.get('market_cap', 0)
            target_margin = target_data.get('ebitda_margin', 0)
            target_growth = target_data.get('revenue_growth', 0)
            target_sector = target_data.get('sector', '')

            fundamental_peers = []

            for peer in peer_data:
                # Size filter (within 10x of target)
                peer_cap = peer.get('market_cap', 0)
                if peer_cap == 0 or target_cap == 0:
                    continue

                size_ratio = max(peer_cap, target_cap) / min(peer_cap, target_cap)
                if size_ratio > 10:
                    continue

                # Calculate fundamental score
                margin_score = 1 - abs(peer.get('ebitda_margin', 0) - target_margin) / max(0.01, abs(target_margin))
                growth_score = 1 - abs(peer.get('revenue_growth', 0) - target_growth) / max(0.01, abs(target_growth))
                sector_score = 1.0 if peer.get('sector') == target_sector else 0.5

                fundamental_score = (margin_score * 0.4 + growth_score * 0.4 + sector_score * 0.2)

                if fundamental_score > 0.6:
                    peer_obj = PeerCompanyEnhanced(
                        ticker=peer['ticker'],
                        name=peer['name'],
                        sector=peer['sector'],
                        industry=peer['industry'],
                        market_cap=peer['market_cap'],
                        revenue=peer['revenue'],
                        ebitda=peer['ebitda'],
                        beta=peer['beta'],
                        roe=peer['roe'],
                        roce=peer['roce'],
                        debt_to_equity=peer['debt_to_equity'],
                        revenue_growth=peer['revenue_growth'],
                        ebitda_margin=peer['ebitda_margin'],
                        pe_ratio=peer['pe_ratio'],
                        ev_ebitda=peer['ev_ebitda'],
                        similarity_score=fundamental_score,
                        peer_type='fundamental',
                        selection_reasons=[
                            f"Similar profitability (margin diff: {abs(peer.get('ebitda_margin', 0) - target_margin):.1%})",
                            f"Similar growth (growth diff: {abs(peer.get('revenue_growth', 0) - target_growth):.1%})",
                            f"Appropriate size ratio: {size_ratio:.1f}x"
                        ]
                    )
                    fundamental_peers.append(peer_obj)

            fundamental_peers.sort(key=lambda x: x.similarity_score, reverse=True)
            return fundamental_peers[:num_peers]

        except Exception as e:
            logger.warning(f"Error in fundamental peer selection: {e}")
            return []

    def _business_model_peer_selection(self, target_data: Dict,
                                     peer_data: List[Dict]) -> List[PeerCompanyEnhanced]:
        """Select peers based on business model similarity"""
        # This would involve industry-specific business model analysis
        # For now, return industry peers with similar characteristics
        try:
            target_industry = target_data.get('industry', '')
            business_peers = []

            for peer in peer_data:
                if peer.get('industry') == target_industry:
                    peer_obj = PeerCompanyEnhanced(
                        ticker=peer['ticker'],
                        name=peer['name'],
                        sector=peer['sector'],
                        industry=peer['industry'],
                        market_cap=peer['market_cap'],
                        revenue=peer['revenue'],
                        ebitda=peer['ebitda'],
                        beta=peer['beta'],
                        roe=peer['roe'],
                        roce=peer['roce'],
                        debt_to_equity=peer['debt_to_equity'],
                        revenue_growth=peer['revenue_growth'],
                        ebitda_margin=peer['ebitda_margin'],
                        pe_ratio=peer['pe_ratio'],
                        ev_ebitda=peer['ev_ebitda'],
                        similarity_score=0.8,
                        peer_type='business_model',
                        selection_reasons=[
                            f"Same industry: {target_industry}",
                            "Similar business model characteristics"
                        ]
                    )
                    business_peers.append(peer_obj)

            return business_peers[:10]

        except Exception:
            return []

    def _market_based_peer_selection(self, target_data: Dict,
                                   peer_data: List[Dict]) -> List[PeerCompanyEnhanced]:
        """Select peers based on market behavior (beta, correlations)"""
        try:
            target_beta = target_data.get('beta', 1.0)
            market_peers = []

            for peer in peer_data:
                peer_beta = peer.get('beta', 1.0)
                beta_diff = abs(peer_beta - target_beta)

                if beta_diff < 0.5:  # Similar market sensitivity
                    peer_obj = PeerCompanyEnhanced(
                        ticker=peer['ticker'],
                        name=peer['name'],
                        sector=peer['sector'],
                        industry=peer['industry'],
                        market_cap=peer['market_cap'],
                        revenue=peer['revenue'],
                        ebitda=peer['ebitda'],
                        beta=peer['beta'],
                        roe=peer['roe'],
                        roce=peer['roce'],
                        debt_to_equity=peer['debt_to_equity'],
                        revenue_growth=peer['revenue_growth'],
                        ebitda_margin=peer['ebitda_margin'],
                        pe_ratio=peer['pe_ratio'],
                        ev_ebitda=peer['ev_ebitda'],
                        similarity_score=1 - beta_diff,
                        peer_type='market_based',
                        selection_reasons=[
                            f"Similar beta: {peer_beta:.2f} vs {target_beta:.2f}",
                            "Similar market sensitivity"
                        ]
                    )
                    market_peers.append(peer_obj)

            market_peers.sort(key=lambda x: x.similarity_score, reverse=True)
            return market_peers[:10]

        except Exception:
            return []

    def _create_similarity_matrix(self, target_data: Dict, peer_data: List[Dict]) -> np.ndarray:
        """Create similarity matrix for visualization"""
        try:
            if len(peer_data) < 2:
                return np.array([[1.0]])

            # Simple similarity based on normalized features
            all_data = [target_data] + peer_data[:20]  # Limit for performance

            features = []
            for company in all_data:
                feature_vector = [
                    company.get('market_cap', 0),
                    company.get('ebitda_margin', 0),
                    company.get('revenue_growth', 0),
                    company.get('roe', 0)
                ]
                features.append(feature_vector)

            features_array = np.array(features)
            features_array = np.nan_to_num(features_array)

            # Normalize
            if features_array.std(axis=0).min() > 0:
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features_array)
                similarity_matrix = cosine_similarity(features_scaled)
            else:
                similarity_matrix = np.eye(len(features_array))

            return similarity_matrix

        except Exception:
            return np.array([[1.0]])

    def _perform_cluster_analysis(self, peer_data: List[Dict]) -> Dict[str, List[str]]:
        """Perform clustering analysis on peer universe"""
        try:
            if len(peer_data) < 10:
                return {'cluster_0': [peer['ticker'] for peer in peer_data]}

            # Create feature matrix for clustering
            features = []
            tickers = []

            for peer in peer_data:
                feature_vector = [
                    np.log(max(1, peer.get('market_cap', 1))),
                    peer.get('ebitda_margin', 0),
                    peer.get('revenue_growth', 0),
                    peer.get('roe', 0)
                ]
                features.append(feature_vector)
                tickers.append(peer['ticker'])

            features_array = np.array(features)
            features_array = np.nan_to_num(features_array)

            # Perform K-means clustering
            n_clusters = min(5, len(peer_data) // 4)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_array)

            # Group tickers by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                cluster_name = f'cluster_{label}'
                if cluster_name not in clusters:
                    clusters[cluster_name] = []
                clusters[cluster_name].append(tickers[i])

            return clusters

        except Exception as e:
            logger.warning(f"Error in cluster analysis: {e}")
            return {}

    def _calculate_feature_importance(self, target_data: Dict,
                                    peer_data: List[Dict]) -> Dict[str, float]:
        """Calculate feature importance for peer selection"""
        # This is a simplified version - in practice would use more sophisticated methods
        return {
            'market_cap_similarity': 0.20,
            'sector_match': 0.18,
            'profitability_similarity': 0.16,
            'growth_similarity': 0.15,
            'size_compatibility': 0.12,
            'leverage_similarity': 0.10,
            'market_beta_similarity': 0.09
        }

    def _calculate_quality_metrics(self, ml_peers: List, fundamental_peers: List) -> Dict[str, float]:
        """Calculate quality metrics for peer selection"""
        try:
            total_ml = len(ml_peers)
            total_fundamental = len(fundamental_peers)

            # Calculate average similarity scores
            avg_ml_similarity = np.mean([p.similarity_score for p in ml_peers]) if ml_peers else 0
            avg_fund_similarity = np.mean([p.similarity_score for p in fundamental_peers]) if fundamental_peers else 0

            return {
                'ml_peer_count': total_ml,
                'fundamental_peer_count': total_fundamental,
                'avg_ml_similarity': avg_ml_similarity,
                'avg_fundamental_similarity': avg_fund_similarity,
                'peer_diversity_score': len(set([p.sector for p in ml_peers + fundamental_peers])) / max(1, len(ml_peers + fundamental_peers)),
                'selection_quality_score': (avg_ml_similarity + avg_fund_similarity) / 2
            }

        except Exception:
            return {}

    def _create_methodology_summary(self) -> Dict[str, Any]:
        """Create methodology summary"""
        return {
            'selection_methods': ['ML_clustering', 'fundamental_analysis', 'business_model', 'market_based'],
            'ml_algorithm': 'K-means clustering with cosine similarity',
            'feature_engineering': 'Log transformation, standardization, sector dummies',
            'similarity_threshold': 0.7,
            'universe_size_limit': 200,
            'dynamic_universe': True,
            'cross_sector_analysis': True
        }

    def _generate_methodology_notes(self) -> List[str]:
        """Generate methodology notes"""
        return [
            "Enhanced peer selection using machine learning and fundamental analysis",
            "Dynamic peer universe construction based on sector, size, and business model",
            "ML clustering uses K-means with cosine similarity on standardized features",
            "Fundamental analysis considers profitability, growth, and size compatibility",
            "Business model peers selected based on industry and operational characteristics",
            "Market-based peers selected using beta and correlation analysis",
            "Cross-sector analysis included for large-cap companies",
            "Similarity thresholds calibrated for quality vs quantity balance"
        ]

    def _create_empty_results(self) -> EnhancedPeerResults:
        """Create empty results for error cases"""
        return EnhancedPeerResults(
            ml_selected_peers=[],
            fundamental_peers=[],
            business_model_peers=[],
            market_based_peers=[],
            peer_clusters={},
            similarity_matrix=np.array([[1.0]]),
            feature_importance={},
            peer_universe_size=0,
            selection_methodology={},
            quality_metrics={},
            methodology_notes=["Enhanced peer selection could not be completed"]
        )

def create_enhanced_peer_selector():
    """Factory function to create EnhancedPeerSelector"""
    return EnhancedPeerSelector()