#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Multi-Agent Cooperation & AI Game Theory Engine
Advanced multi-agent cooperation with voting, argumentation, and reinforcement learning
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
from abc import ABC, abstractmethod

class AgentType(Enum):
    """Types of trading agents"""
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    ML_PREDICTOR = "ml_predictor"
    WHALE_DETECTOR = "whale_detector"
    BACKTEST = "backtest"
    TRADE_EXECUTOR = "trade_executor"

class DecisionWeight(Enum):
    """Agent decision weights based on expertise"""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9

@dataclass
class AgentDecision:
    """Individual agent decision"""
    agent_type: AgentType
    symbol: str
    decision: str  # BUY/SELL/HOLD
    confidence: float
    reasoning: List[str]
    supporting_data: Dict[str, Any]
    timestamp: datetime
    weight: float = 0.5
    
@dataclass
class ConsensusResult:
    """Result of multi-agent consensus"""
    final_decision: str
    confidence: float
    participating_agents: List[AgentType]
    consensus_score: float
    disagreement_level: float
    majority_reasoning: List[str]
    dissenting_opinions: List[Dict[str, Any]]

class AgentArgumentationEngine:
    """Advanced argumentation system for agent cooperation"""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.argument_history = []
        
    def conduct_agent_debate(self, decisions: List[AgentDecision], 
                           symbol: str) -> Dict[str, Any]:
        """Conduct structured debate between agents"""
        try:
            # Group decisions by type
            buy_agents = [d for d in decisions if d.decision == 'BUY']
            sell_agents = [d for d in decisions if d.decision == 'SELL']
            hold_agents = [d for d in decisions if d.decision == 'HOLD']
            
            debate_result = {
                'symbol': symbol,
                'debate_rounds': [],
                'final_arguments': {},
                'consensus_reached': False,
                'timestamp': datetime.now().isoformat()
            }
            
            # Round 1: Initial positions
            initial_round = self._conduct_debate_round(
                "Initial Positions", buy_agents, sell_agents, hold_agents
            )
            debate_result['debate_rounds'].append(initial_round)
            
            # Round 2: Counter-arguments
            counter_round = self._conduct_counter_arguments(
                buy_agents, sell_agents, hold_agents, initial_round
            )
            debate_result['debate_rounds'].append(counter_round)
            
            # Round 3: Evidence weighing
            evidence_round = self._weigh_evidence(decisions)
            debate_result['debate_rounds'].append(evidence_round)
            
            # Final consensus attempt
            consensus = self._attempt_consensus(decisions, debate_result['debate_rounds'])
            debate_result['final_arguments'] = consensus
            debate_result['consensus_reached'] = consensus['consensus_achieved']
            
            self.argument_history.append(debate_result)
            
            return debate_result
            
        except Exception as e:
            self.logger.error(f"Agent debate failed: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _conduct_debate_round(self, round_name: str, buy_agents: List[AgentDecision],
                            sell_agents: List[AgentDecision], 
                            hold_agents: List[AgentDecision]) -> Dict[str, Any]:
        """Conduct a single debate round"""
        try:
            round_result = {
                'round_name': round_name,
                'buy_arguments': [],
                'sell_arguments': [],
                'hold_arguments': [],
                'strongest_points': {}
            }
            
            # Collect buy arguments
            for agent in buy_agents:
                argument = {
                    'agent': agent.agent_type.value,
                    'confidence': agent.confidence,
                    'reasoning': agent.reasoning,
                    'key_data': self._extract_key_data(agent.supporting_data),
                    'argument_strength': self._calculate_argument_strength(agent)
                }
                round_result['buy_arguments'].append(argument)
            
            # Collect sell arguments
            for agent in sell_agents:
                argument = {
                    'agent': agent.agent_type.value,
                    'confidence': agent.confidence,
                    'reasoning': agent.reasoning,
                    'key_data': self._extract_key_data(agent.supporting_data),
                    'argument_strength': self._calculate_argument_strength(agent)
                }
                round_result['sell_arguments'].append(argument)
            
            # Collect hold arguments
            for agent in hold_agents:
                argument = {
                    'agent': agent.agent_type.value,
                    'confidence': agent.confidence,
                    'reasoning': agent.reasoning,
                    'key_data': self._extract_key_data(agent.supporting_data),
                    'argument_strength': self._calculate_argument_strength(agent)
                }
                round_result['hold_arguments'].append(argument)
            
            # Identify strongest points
            round_result['strongest_points'] = self._identify_strongest_points(round_result)
            
            return round_result
            
        except Exception as e:
            self.logger.warning(f"Debate round failed: {e}")
            return {'error': str(e)}
    
    def _conduct_counter_arguments(self, buy_agents: List[AgentDecision],
                                 sell_agents: List[AgentDecision],
                                 hold_agents: List[AgentDecision],
                                 initial_round: Dict[str, Any]) -> Dict[str, Any]:
        """Generate counter-arguments between opposing sides"""
        try:
            counter_round = {
                'round_name': 'Counter-Arguments',
                'buy_counters': [],
                'sell_counters': [],
                'conflicts_identified': [],
                'resolution_attempts': []
            }
            
            # Buy agents counter sell arguments
            strongest_sell_args = initial_round.get('strongest_points', {}).get('sell', [])
            for sell_arg in strongest_sell_args[:2]:  # Top 2 sell arguments
                for buy_agent in buy_agents:
                    counter = self._generate_counter_argument(buy_agent, sell_arg, 'BUY')
                    if counter:
                        counter_round['buy_counters'].append(counter)
            
            # Sell agents counter buy arguments
            strongest_buy_args = initial_round.get('strongest_points', {}).get('buy', [])
            for buy_arg in strongest_buy_args[:2]:  # Top 2 buy arguments
                for sell_agent in sell_agents:
                    counter = self._generate_counter_argument(sell_agent, buy_arg, 'SELL')
                    if counter:
                        counter_round['sell_counters'].append(counter)
            
            # Identify key conflicts
            conflicts = self._identify_conflicts(buy_agents, sell_agents)
            counter_round['conflicts_identified'] = conflicts
            
            return counter_round
            
        except Exception as e:
            self.logger.warning(f"Counter-argument generation failed: {e}")
            return {'error': str(e)}
    
    def _weigh_evidence(self, decisions: List[AgentDecision]) -> Dict[str, Any]:
        """Weigh evidence presented by all agents"""
        try:
            evidence_round = {
                'round_name': 'Evidence Weighing',
                'evidence_categories': {},
                'data_quality_scores': {},
                'reliability_assessment': {}
            }
            
            # Categorize evidence by type
            evidence_categories = {
                'technical_indicators': [],
                'sentiment_data': [],
                'volume_analysis': [],
                'price_action': [],
                'external_factors': []
            }
            
            for decision in decisions:
                agent_evidence = self._categorize_evidence(decision)
                for category, evidence in agent_evidence.items():
                    if evidence:
                        evidence_categories[category].extend(evidence)
            
            evidence_round['evidence_categories'] = evidence_categories
            
            # Score data quality
            for category, evidence_list in evidence_categories.items():
                quality_score = self._assess_data_quality(evidence_list)
                evidence_round['data_quality_scores'][category] = quality_score
            
            # Assess agent reliability
            for decision in decisions:
                reliability = self._assess_agent_reliability(decision)
                evidence_round['reliability_assessment'][decision.agent_type.value] = reliability
            
            return evidence_round
            
        except Exception as e:
            self.logger.warning(f"Evidence weighing failed: {e}")
            return {'error': str(e)}
    
    def _attempt_consensus(self, decisions: List[AgentDecision], 
                         debate_rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Attempt to reach consensus based on debate"""
        try:
            consensus_result = {
                'consensus_achieved': False,
                'final_decision': 'HOLD',
                'confidence_level': 0.5,
                'supporting_evidence': [],
                'dissenting_views': [],
                'decision_rationale': []
            }
            
            # Weight decisions based on debate performance
            weighted_decisions = self._apply_debate_weights(decisions, debate_rounds)
            
            # Calculate weighted vote
            buy_weight = sum(d['weight'] for d in weighted_decisions if d['decision'] == 'BUY')
            sell_weight = sum(d['weight'] for d in weighted_decisions if d['decision'] == 'SELL')
            hold_weight = sum(d['weight'] for d in weighted_decisions if d['decision'] == 'HOLD')
            
            total_weight = buy_weight + sell_weight + hold_weight
            
            if total_weight > 0:
                buy_pct = buy_weight / total_weight
                sell_pct = sell_weight / total_weight
                hold_pct = hold_weight / total_weight
                
                # Determine consensus
                if buy_pct > 0.6:
                    consensus_result['final_decision'] = 'BUY'
                    consensus_result['confidence_level'] = buy_pct
                    consensus_result['consensus_achieved'] = True
                elif sell_pct > 0.6:
                    consensus_result['final_decision'] = 'SELL'
                    consensus_result['confidence_level'] = sell_pct
                    consensus_result['consensus_achieved'] = True
                elif max(buy_pct, sell_pct, hold_pct) > 0.5:
                    # Weak consensus
                    if buy_pct > sell_pct and buy_pct > hold_pct:
                        consensus_result['final_decision'] = 'BUY'
                        consensus_result['confidence_level'] = buy_pct
                    elif sell_pct > hold_pct:
                        consensus_result['final_decision'] = 'SELL'
                        consensus_result['confidence_level'] = sell_pct
                    else:
                        consensus_result['final_decision'] = 'HOLD'
                        consensus_result['confidence_level'] = hold_pct
                
                # Collect supporting evidence for final decision
                supporting_agents = [d for d in weighted_decisions 
                                   if d['decision'] == consensus_result['final_decision']]
                consensus_result['supporting_evidence'] = [
                    {'agent': agent['agent_type'], 'reasoning': agent.get('reasoning', [])}
                    for agent in supporting_agents
                ]
                
                # Collect dissenting views
                dissenting_agents = [d for d in weighted_decisions 
                                   if d['decision'] != consensus_result['final_decision']]
                consensus_result['dissenting_views'] = [
                    {'agent': agent['agent_type'], 'decision': agent['decision'], 
                     'reasoning': agent.get('reasoning', [])}
                    for agent in dissenting_agents
                ]
            
            return consensus_result
            
        except Exception as e:
            self.logger.warning(f"Consensus building failed: {e}")
            return {'error': str(e)}
    
    def _extract_key_data(self, supporting_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key data points from supporting data"""
        try:
            key_data = {}
            
            # Extract numerical indicators
            numerical_keys = ['rsi', 'volume_ratio', 'price_change', 'sentiment_score', 'confidence']
            for key in numerical_keys:
                if key in supporting_data:
                    key_data[key] = supporting_data[key]
            
            # Extract categorical data
            categorical_keys = ['trend_direction', 'market_regime', 'signal_type']
            for key in categorical_keys:
                if key in supporting_data:
                    key_data[key] = supporting_data[key]
            
            return key_data
            
        except Exception:
            return {}
    
    def _calculate_argument_strength(self, agent_decision: AgentDecision) -> float:
        """Calculate strength of an agent's argument"""
        try:
            # Base strength from confidence
            strength = agent_decision.confidence
            
            # Boost for detailed reasoning
            if len(agent_decision.reasoning) > 2:
                strength += 0.1
            
            # Boost for supporting data
            if len(agent_decision.supporting_data) > 3:
                strength += 0.1
            
            # Agent type specialization bonus
            specialization_bonus = {
                AgentType.TECHNICAL: 0.1,
                AgentType.ML_PREDICTOR: 0.15,
                AgentType.SENTIMENT: 0.05,
                AgentType.WHALE_DETECTOR: 0.05
            }
            
            strength += specialization_bonus.get(agent_decision.agent_type, 0.0)
            
            return min(1.0, strength)
            
        except Exception:
            return 0.5
    
    def _identify_strongest_points(self, round_result: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Identify strongest arguments from each side"""
        try:
            strongest_points = {'buy': [], 'sell': [], 'hold': []}
            
            # Sort buy arguments by strength
            buy_args = round_result.get('buy_arguments', [])
            buy_args_sorted = sorted(buy_args, key=lambda x: x.get('argument_strength', 0), reverse=True)
            strongest_points['buy'] = buy_args_sorted[:2]  # Top 2
            
            # Sort sell arguments by strength
            sell_args = round_result.get('sell_arguments', [])
            sell_args_sorted = sorted(sell_args, key=lambda x: x.get('argument_strength', 0), reverse=True)
            strongest_points['sell'] = sell_args_sorted[:2]  # Top 2
            
            # Sort hold arguments by strength
            hold_args = round_result.get('hold_arguments', [])
            hold_args_sorted = sorted(hold_args, key=lambda x: x.get('argument_strength', 0), reverse=True)
            strongest_points['hold'] = hold_args_sorted[:1]  # Top 1
            
            return strongest_points
            
        except Exception:
            return {'buy': [], 'sell': [], 'hold': []}
    
    def _generate_counter_argument(self, agent: AgentDecision, 
                                 opposing_arg: Dict[str, Any], 
                                 position: str) -> Optional[Dict[str, Any]]:
        """Generate counter-argument from agent against opposing view"""
        try:
            # Simplified counter-argument generation
            counter_points = []
            
            # Counter based on agent's specialty
            if agent.agent_type == AgentType.TECHNICAL:
                if 'rsi' in agent.supporting_data:
                    rsi = agent.supporting_data['rsi']
                    if position == 'BUY' and rsi < 30:
                        counter_points.append(f"Technical indicators show RSI oversold at {rsi:.1f}")
                    elif position == 'SELL' and rsi > 70:
                        counter_points.append(f"Technical indicators show RSI overbought at {rsi:.1f}")
            
            elif agent.agent_type == AgentType.SENTIMENT:
                if 'sentiment_score' in agent.supporting_data:
                    sentiment = agent.supporting_data['sentiment_score']
                    if position == 'BUY' and sentiment > 0.6:
                        counter_points.append(f"Sentiment analysis shows positive outlook ({sentiment:.2f})")
                    elif position == 'SELL' and sentiment < 0.4:
                        counter_points.append(f"Sentiment analysis shows negative outlook ({sentiment:.2f})")
            
            if counter_points:
                return {
                    'agent': agent.agent_type.value,
                    'position': position,
                    'counter_points': counter_points,
                    'target_argument': opposing_arg.get('agent', 'unknown'),
                    'strength': len(counter_points) * 0.3
                }
            
            return None
            
        except Exception:
            return None
    
    def _identify_conflicts(self, buy_agents: List[AgentDecision], 
                          sell_agents: List[AgentDecision]) -> List[Dict[str, Any]]:
        """Identify key conflicts between buy and sell positions"""
        try:
            conflicts = []
            
            # Price direction conflict
            if buy_agents and sell_agents:
                conflicts.append({
                    'type': 'price_direction',
                    'description': 'Fundamental disagreement on price direction',
                    'buy_agents': [agent.agent_type.value for agent in buy_agents],
                    'sell_agents': [agent.agent_type.value for agent in sell_agents]
                })
            
            # Technical vs Sentiment conflict
            technical_agents = [agent for agent in buy_agents + sell_agents 
                              if agent.agent_type == AgentType.TECHNICAL]
            sentiment_agents = [agent for agent in buy_agents + sell_agents 
                              if agent.agent_type == AgentType.SENTIMENT]
            
            if technical_agents and sentiment_agents:
                tech_decisions = [agent.decision for agent in technical_agents]
                sent_decisions = [agent.decision for agent in sentiment_agents]
                
                if set(tech_decisions) != set(sent_decisions):
                    conflicts.append({
                        'type': 'technical_vs_sentiment',
                        'description': 'Technical analysis conflicts with sentiment analysis',
                        'technical_view': tech_decisions[0] if tech_decisions else 'UNKNOWN',
                        'sentiment_view': sent_decisions[0] if sent_decisions else 'UNKNOWN'
                    })
            
            return conflicts
            
        except Exception:
            return []
    
    def _categorize_evidence(self, decision: AgentDecision) -> Dict[str, List[str]]:
        """Categorize evidence by type"""
        try:
            categorized = {
                'technical_indicators': [],
                'sentiment_data': [],
                'volume_analysis': [],
                'price_action': [],
                'external_factors': []
            }
            
            supporting_data = decision.supporting_data
            
            # Technical indicators
            technical_keys = ['rsi', 'macd', 'bollinger_bands', 'moving_averages']
            for key in technical_keys:
                if key in supporting_data:
                    categorized['technical_indicators'].append(f"{key}: {supporting_data[key]}")
            
            # Sentiment data
            sentiment_keys = ['sentiment_score', 'social_volume', 'news_sentiment']
            for key in sentiment_keys:
                if key in supporting_data:
                    categorized['sentiment_data'].append(f"{key}: {supporting_data[key]}")
            
            # Volume analysis
            volume_keys = ['volume_ratio', 'volume_trend', 'unusual_volume']
            for key in volume_keys:
                if key in supporting_data:
                    categorized['volume_analysis'].append(f"{key}: {supporting_data[key]}")
            
            # Price action
            price_keys = ['price_change', 'trend_direction', 'support_resistance']
            for key in price_keys:
                if key in supporting_data:
                    categorized['price_action'].append(f"{key}: {supporting_data[key]}")
            
            return categorized
            
        except Exception:
            return {'technical_indicators': [], 'sentiment_data': [], 'volume_analysis': [], 'price_action': [], 'external_factors': []}
    
    def _assess_data_quality(self, evidence_list: List[str]) -> float:
        """Assess quality of evidence"""
        try:
            if not evidence_list:
                return 0.0
            
            # Simple quality scoring based on evidence count and diversity
            quality_score = min(1.0, len(evidence_list) / 5.0)  # Up to 5 pieces of evidence
            
            return quality_score
            
        except Exception:
            return 0.5
    
    def _assess_agent_reliability(self, decision: AgentDecision) -> Dict[str, float]:
        """Assess reliability of agent decision"""
        try:
            reliability = {
                'confidence_score': decision.confidence,
                'reasoning_depth': min(1.0, len(decision.reasoning) / 3.0),
                'data_support': min(1.0, len(decision.supporting_data) / 5.0)
            }
            
            reliability['overall'] = sum(reliability.values()) / len(reliability)
            
            return reliability
            
        except Exception:
            return {'confidence_score': 0.5, 'reasoning_depth': 0.5, 'data_support': 0.5, 'overall': 0.5}
    
    def _apply_debate_weights(self, decisions: List[AgentDecision], 
                            debate_rounds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply weights based on debate performance"""
        try:
            weighted_decisions = []
            
            for decision in decisions:
                # Start with base weight
                weight = decision.weight
                
                # Boost for strong arguments in debate
                for round_data in debate_rounds:
                    if round_data.get('round_name') == 'Initial Positions':
                        strongest_points = round_data.get('strongest_points', {})
                        
                        # Check if this agent had strong arguments
                        for position, points in strongest_points.items():
                            for point in points:
                                if point.get('agent') == decision.agent_type.value:
                                    weight += 0.1  # Boost for strong argument
                
                weighted_decisions.append({
                    'agent_type': decision.agent_type.value,
                    'decision': decision.decision,
                    'weight': min(1.0, weight),
                    'reasoning': decision.reasoning
                })
            
            return weighted_decisions
            
        except Exception:
            # Fallback to original decisions
            return [
                {
                    'agent_type': d.agent_type.value,
                    'decision': d.decision,
                    'weight': d.weight,
                    'reasoning': d.reasoning
                }
                for d in decisions
            ]

class VotingSystem:
    """Advanced voting system with multiple algorithms"""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
    
    def conduct_weighted_vote(self, decisions: List[AgentDecision]) -> ConsensusResult:
        """Conduct weighted voting among agents"""
        try:
            if not decisions:
                return self._create_empty_consensus()
            
            # Calculate weighted votes
            vote_weights = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
            agent_contributions = []
            
            for decision in decisions:
                weight = self._calculate_agent_weight(decision)
                vote_weights[decision.decision] += weight * decision.confidence
                
                agent_contributions.append({
                    'agent': decision.agent_type,
                    'decision': decision.decision,
                    'weight': weight,
                    'confidence': decision.confidence,
                    'contribution': weight * decision.confidence
                })
            
            # Determine winner
            total_weight = sum(vote_weights.values())
            if total_weight == 0:
                return self._create_empty_consensus()
            
            # Normalize weights
            normalized_weights = {k: v/total_weight for k, v in vote_weights.items()}
            
            # Find winning decision
            winning_decision = max(normalized_weights.keys(), key=lambda k: normalized_weights[k])
            winning_confidence = normalized_weights[winning_decision]
            
            # Calculate consensus metrics
            consensus_score = self._calculate_consensus_score(normalized_weights)
            disagreement_level = 1.0 - consensus_score
            
            # Collect reasoning
            majority_reasoning = []
            dissenting_opinions = []
            
            for decision in decisions:
                if decision.decision == winning_decision:
                    majority_reasoning.extend(decision.reasoning)
                else:
                    dissenting_opinions.append({
                        'agent': decision.agent_type.value,
                        'decision': decision.decision,
                        'reasoning': decision.reasoning,
                        'confidence': decision.confidence
                    })
            
            return ConsensusResult(
                final_decision=winning_decision,
                confidence=winning_confidence,
                participating_agents=[d.agent_type for d in decisions],
                consensus_score=consensus_score,
                disagreement_level=disagreement_level,
                majority_reasoning=majority_reasoning[:5],  # Top 5 reasons
                dissenting_opinions=dissenting_opinions
            )
            
        except Exception as e:
            self.logger.error(f"Weighted voting failed: {e}")
            return self._create_empty_consensus()
    
    def conduct_borda_count(self, decisions: List[AgentDecision]) -> Dict[str, Any]:
        """Conduct Borda count voting (ranked choice)"""
        try:
            if len(decisions) < 2:
                return {'error': 'insufficient_decisions'}
            
            # Create preference rankings for each agent
            rankings = {}
            for decision in decisions:
                agent_id = decision.agent_type.value
                
                # Create ranking based on confidence
                if decision.decision == 'BUY':
                    if decision.confidence > 0.7:
                        rankings[agent_id] = ['BUY', 'HOLD', 'SELL']
                    else:
                        rankings[agent_id] = ['BUY', 'SELL', 'HOLD']
                elif decision.decision == 'SELL':
                    if decision.confidence > 0.7:
                        rankings[agent_id] = ['SELL', 'HOLD', 'BUY']
                    else:
                        rankings[agent_id] = ['SELL', 'BUY', 'HOLD']
                else:  # HOLD
                    rankings[agent_id] = ['HOLD', 'BUY', 'SELL']
            
            # Calculate Borda scores
            borda_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            
            for agent_id, ranking in rankings.items():
                for i, choice in enumerate(ranking):
                    # Award points: 1st place = 2 points, 2nd = 1 point, 3rd = 0 points
                    points = max(0, len(ranking) - 1 - i)
                    borda_scores[choice] += points
            
            # Determine winner
            winner = max(borda_scores.keys(), key=lambda k: borda_scores[k])
            
            return {
                'method': 'borda_count',
                'winner': winner,
                'scores': borda_scores,
                'rankings': rankings,
                'total_agents': len(decisions)
            }
            
        except Exception as e:
            self.logger.error(f"Borda count failed: {e}")
            return {'error': str(e)}
    
    def _calculate_agent_weight(self, decision: AgentDecision) -> float:
        """Calculate weight for agent based on type and performance"""
        try:
            # Base weights by agent type
            base_weights = {
                AgentType.ML_PREDICTOR: 0.25,
                AgentType.TECHNICAL: 0.20,
                AgentType.SENTIMENT: 0.15,
                AgentType.WHALE_DETECTOR: 0.15,
                AgentType.BACKTEST: 0.15,
                AgentType.TRADE_EXECUTOR: 0.10
            }
            
            base_weight = base_weights.get(decision.agent_type, 0.15)
            
            # Adjust based on decision quality
            if len(decision.reasoning) > 2:
                base_weight *= 1.1
            
            if len(decision.supporting_data) > 3:
                base_weight *= 1.1
            
            return min(1.0, base_weight)
            
        except Exception:
            return 0.15
    
    def _calculate_consensus_score(self, normalized_weights: Dict[str, float]) -> float:
        """Calculate consensus score based on weight distribution"""
        try:
            # Consensus is higher when one option dominates
            max_weight = max(normalized_weights.values())
            
            # Calculate entropy (lower entropy = higher consensus)
            entropy = 0.0
            for weight in normalized_weights.values():
                if weight > 0:
                    entropy -= weight * np.log2(weight)
            
            # Normalize entropy to 0-1 scale
            max_entropy = np.log2(len(normalized_weights))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            # Consensus score (1 - normalized_entropy)
            consensus_score = 1.0 - normalized_entropy
            
            return max(0.0, min(1.0, consensus_score))
            
        except Exception:
            return 0.5
    
    def _create_empty_consensus(self) -> ConsensusResult:
        """Create empty consensus result"""
        return ConsensusResult(
            final_decision='HOLD',
            confidence=0.5,
            participating_agents=[],
            consensus_score=0.0,
            disagreement_level=1.0,
            majority_reasoning=['No clear consensus'],
            dissenting_opinions=[]
        )

class MultiAgentCooperationCoordinator:
    """Main coordinator for multi-agent cooperation and game theory"""
    
    def __init__(self, config_manager=None, cache_manager=None):
        self.config_manager = config_manager
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.argumentation_engine = AgentArgumentationEngine(config_manager)
        self.voting_system = VotingSystem(config_manager)
        
        # Cooperation history
        self.cooperation_history = []
        self.agent_performance_tracking = {}
        
        self.logger.info("Multi-Agent Cooperation Coordinator initialized")
    
    def conduct_comprehensive_cooperation(self, agent_decisions: List[Dict[str, Any]], 
                                        symbol: str) -> Dict[str, Any]:
        """Conduct comprehensive multi-agent cooperation process"""
        try:
            # Convert to AgentDecision objects
            decisions = []
            for decision_data in agent_decisions:
                try:
                    agent_decision = AgentDecision(
                        agent_type=AgentType(decision_data.get('agent_type', 'sentiment')),
                        symbol=symbol,
                        decision=decision_data.get('decision', 'HOLD'),
                        confidence=decision_data.get('confidence', 0.5),
                        reasoning=decision_data.get('reasoning', []),
                        supporting_data=decision_data.get('supporting_data', {}),
                        timestamp=datetime.now(),
                        weight=decision_data.get('weight', 0.5)
                    )
                    decisions.append(agent_decision)
                except Exception as e:
                    self.logger.warning(f"Failed to create AgentDecision: {e}")
                    continue
            
            if not decisions:
                return {'error': 'no_valid_decisions'}
            
            cooperation_result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'participating_agents': [d.agent_type.value for d in decisions],
                'cooperation_methods': []
            }
            
            # Method 1: Weighted Voting
            voting_result = self.voting_system.conduct_weighted_vote(decisions)
            cooperation_result['weighted_voting'] = asdict(voting_result)
            cooperation_result['cooperation_methods'].append('weighted_voting')
            
            # Method 2: Borda Count (if enough agents)
            if len(decisions) >= 3:
                borda_result = self.voting_system.conduct_borda_count(decisions)
                cooperation_result['borda_count'] = borda_result
                cooperation_result['cooperation_methods'].append('borda_count')
            
            # Method 3: Agent Argumentation (if disagreement)
            if voting_result.disagreement_level > 0.3:
                debate_result = self.argumentation_engine.conduct_agent_debate(decisions, symbol)
                cooperation_result['agent_debate'] = debate_result
                cooperation_result['cooperation_methods'].append('agent_debate')
            
            # Final recommendation
            final_recommendation = self._synthesize_cooperation_results(cooperation_result)
            cooperation_result['final_recommendation'] = final_recommendation
            
            # Track performance
            self._track_cooperation_performance(cooperation_result)
            
            # Store in history
            self.cooperation_history.append(cooperation_result)
            
            return cooperation_result
            
        except Exception as e:
            self.logger.error(f"Comprehensive cooperation failed: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _synthesize_cooperation_results(self, cooperation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from multiple cooperation methods"""
        try:
            synthesis = {
                'recommended_action': 'HOLD',
                'confidence': 0.5,
                'synthesis_method': 'weighted_average',
                'supporting_methods': [],
                'risk_assessment': 'moderate'
            }
            
            # Collect decisions from different methods
            method_decisions = []
            
            # Weighted voting result
            if 'weighted_voting' in cooperation_result:
                wv = cooperation_result['weighted_voting']
                method_decisions.append({
                    'method': 'weighted_voting',
                    'decision': wv['final_decision'],
                    'confidence': wv['confidence'],
                    'weight': 0.4  # High weight for weighted voting
                })
            
            # Borda count result
            if 'borda_count' in cooperation_result:
                bc = cooperation_result['borda_count']
                if 'winner' in bc:
                    # Calculate confidence from scores
                    total_score = sum(bc['scores'].values())
                    winner_score = bc['scores'][bc['winner']]
                    borda_confidence = winner_score / total_score if total_score > 0 else 0.5
                    
                    method_decisions.append({
                        'method': 'borda_count',
                        'decision': bc['winner'],
                        'confidence': borda_confidence,
                        'weight': 0.3
                    })
            
            # Debate result
            if 'agent_debate' in cooperation_result:
                debate = cooperation_result['agent_debate']
                if 'final_arguments' in debate and 'final_decision' in debate['final_arguments']:
                    method_decisions.append({
                        'method': 'agent_debate',
                        'decision': debate['final_arguments']['final_decision'],
                        'confidence': debate['final_arguments'].get('confidence_level', 0.5),
                        'weight': 0.3
                    })
            
            # Synthesize final decision
            if method_decisions:
                # Weight decisions
                decision_weights = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
                total_weight = 0.0
                
                for method_result in method_decisions:
                    weight = method_result['weight'] * method_result['confidence']
                    decision_weights[method_result['decision']] += weight
                    total_weight += weight
                    synthesis['supporting_methods'].append(method_result['method'])
                
                if total_weight > 0:
                    # Normalize and find winner
                    normalized_weights = {k: v/total_weight for k, v in decision_weights.items()}
                    synthesis['recommended_action'] = max(normalized_weights.keys(), 
                                                        key=lambda k: normalized_weights[k])
                    synthesis['confidence'] = normalized_weights[synthesis['recommended_action']]
            
            # Risk assessment
            if synthesis['confidence'] > 0.7:
                synthesis['risk_assessment'] = 'low'
            elif synthesis['confidence'] < 0.4:
                synthesis['risk_assessment'] = 'high'
            else:
                synthesis['risk_assessment'] = 'moderate'
            
            return synthesis
            
        except Exception as e:
            self.logger.warning(f"Synthesis failed: {e}")
            return {
                'recommended_action': 'HOLD',
                'confidence': 0.5,
                'synthesis_method': 'fallback',
                'error': str(e)
            }
    
    def _track_cooperation_performance(self, cooperation_result: Dict[str, Any]):
        """Track performance of cooperation methods"""
        try:
            timestamp = cooperation_result.get('timestamp', datetime.now().isoformat())
            
            # Track by method
            for method in cooperation_result.get('cooperation_methods', []):
                if method not in self.agent_performance_tracking:
                    self.agent_performance_tracking[method] = {
                        'total_uses': 0,
                        'decisions': {'BUY': 0, 'SELL': 0, 'HOLD': 0},
                        'avg_confidence': 0.0,
                        'last_used': timestamp
                    }
                
                tracking = self.agent_performance_tracking[method]
                tracking['total_uses'] += 1
                tracking['last_used'] = timestamp
                
                # Track decision if available
                if method == 'weighted_voting' and 'weighted_voting' in cooperation_result:
                    decision = cooperation_result['weighted_voting']['final_decision']
                    confidence = cooperation_result['weighted_voting']['confidence']
                    tracking['decisions'][decision] += 1
                    tracking['avg_confidence'] = (tracking['avg_confidence'] + confidence) / 2
                
        except Exception as e:
            self.logger.warning(f"Performance tracking failed: {e}")
    
    def get_cooperation_report(self) -> Dict[str, Any]:
        """Generate comprehensive cooperation report"""
        try:
            return {
                'total_cooperation_sessions': len(self.cooperation_history),
                'method_performance': self.agent_performance_tracking,
                'recent_sessions': self.cooperation_history[-5:] if self.cooperation_history else [],
                'argumentation_history_count': len(self.argumentation_engine.argument_history),
                'system_status': {
                    'argumentation_engine_active': self.argumentation_engine is not None,
                    'voting_system_active': self.voting_system is not None
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

# Convenience function
def get_multi_agent_coordinator(config_manager=None, cache_manager=None) -> MultiAgentCooperationCoordinator:
    """Get configured multi-agent cooperation coordinator"""
    return MultiAgentCooperationCoordinator(config_manager, cache_manager)

if __name__ == "__main__":
    # Test the multi-agent cooperation engine
    coordinator = get_multi_agent_coordinator()
    
    print("Testing Multi-Agent Cooperation Engine...")
    
    # Test agent decisions
    test_decisions = [
        {
            'agent_type': 'technical',
            'decision': 'BUY',
            'confidence': 0.8,
            'reasoning': ['RSI oversold', 'Support level hold'],
            'supporting_data': {'rsi': 25, 'support': 45000},
            'weight': 0.7
        },
        {
            'agent_type': 'sentiment',
            'decision': 'SELL',
            'confidence': 0.6,
            'reasoning': ['Negative news sentiment'],
            'supporting_data': {'sentiment_score': 0.3},
            'weight': 0.5
        },
        {
            'agent_type': 'ml_predictor',
            'decision': 'BUY',
            'confidence': 0.75,
            'reasoning': ['Model prediction bullish'],
            'supporting_data': {'prediction_confidence': 0.75},
            'weight': 0.8
        }
    ]
    
    # Test cooperation
    result = coordinator.conduct_comprehensive_cooperation(test_decisions, 'BTC/USD')
    
    print(f"\nCooperation Result:")
    print(f"  Final Recommendation: {result['final_recommendation']['recommended_action']}")
    print(f"  Confidence: {result['final_recommendation']['confidence']:.2f}")
    print(f"  Methods Used: {result['cooperation_methods']}")
    print(f"  Risk Assessment: {result['final_recommendation']['risk_assessment']}")
    
    # Get report
    report = coordinator.get_cooperation_report()
    print(f"\nCooperation Report:")
    print(f"  Total Sessions: {report['total_cooperation_sessions']}")
    print(f"  Active Components: {report['system_status']}")
    
    print("Multi-agent cooperation test completed")