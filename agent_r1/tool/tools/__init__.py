"""
Specific tool implementations
"""

from agent_r1.tool.tools.search_tool import SearchTool
from agent_r1.tool.tools.calculator_tool import CalculatorTool
from agent_r1.tool.tools.wiki_search_tool import WikiSearchTool
from agent_r1.tool.tools.mathsearch_tool import MathSearchTool
__all__ = [
    'SearchTool',
    'CalculatorTool',
    'WikiSearchTool',
    'MathSearchTool'
] 

def _default_tools(env):
    if env == 'search':
        return [SearchTool()]
    if env == 'mathsearch':
        return [MathSearchTool()]
    elif env == 'calculator':
        return [CalculatorTool()]
    elif env == 'wikisearch':
        return [WikiSearchTool()]
    else:
        raise NotImplementedError
