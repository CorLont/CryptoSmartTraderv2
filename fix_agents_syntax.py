#!/usr/bin/env python3
import os
import re

# Direct fixes for specific agent files
agent_fixes = {
    'src/cryptosmarttrader/agents/agents/scraping_core/data_sources.py': [
        ('return self._generate_mock_datasymbol, limit)', 'return self._generate_mock_data(symbol, limit)')
    ],
    'src/cryptosmarttrader/agents/agents/scraping_core/async_client.py': [
        ('async def fetch_json(self, \n', 'async def fetch_json(self,')
    ],
    'src/cryptosmarttrader/agents/agents/enhanced_ml_agent.py': [
        ('    return df\n    ^\n', '    return df'),
        ('\n    return df', '\n        return df')
    ],
    'src/cryptosmarttrader/agents/agents/enhanced_sentiment_agent.py': [
        ('result = SentimentResult(\n                            ^', 'result = SentimentResult()'),
        ('\n        result = SentimentResult(', '\n        result = SentimentResult()')
    ],
    'src/cryptosmarttrader/agents/agents/ensemble_voting_agent.py': [
        ('position_size = max(\n                       ^', 'position_size = max(0.0, 1.0)')
    ],
    'src/cryptosmarttrader/agents/agents/listing_detection_agent.py': [
        ('}\n    ^', '}')
    ],
    'src/cryptosmarttrader/agents/agents/ml_predictor_agent.py': [
        ('interval = (\n               ^', 'interval = 60')
    ],
    'src/cryptosmarttrader/agents/agents/news_speed_agent.py': [
        ('self.session = aiohttp.ClientSession(\n                                        ^', 'self.session = aiohttp.ClientSession()')
    ],
    'src/cryptosmarttrader/agents/agents/portfolio_optimizer_agent.py': [
        ('weights, np.dot(covariance_matrix', 'return weights, np.dot(covariance_matrix')
    ],
    'src/cryptosmarttrader/agents/agents/sentiment_agent.py': [
        ('sentiment_data = SentimentData(\n                                  ^', 'sentiment_data = SentimentData()')
    ],
    'src/cryptosmarttrader/agents/agents/trade_executor_agent.py': [
        ('self.config_manager.get("agents", {})\n    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^', 'return self.config_manager.get("agents", {})')
    ]
}

def fix_agent_files():
    for file_path, fixes in agent_fixes.items():
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            original_content = content
            for old, new in fixes:
                content = content.replace(old, new)
            
            # Specific regex patterns for more complex issues
            content = re.sub(r'return self\._generate_mock_data([^)]*symbol[^)]*limit[^)]*)\)', r'return self._generate_mock_data(symbol, limit)', content)
            
            if content != original_content:
                with open(file_path, 'w') as f:
                    f.write(content)
                print(f'Fixed: {file_path}')

if __name__ == "__main__":
    fix_agent_files()
    print("Agent-specific syntax fixes completed")
