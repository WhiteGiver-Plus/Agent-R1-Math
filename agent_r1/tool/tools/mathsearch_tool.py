"""
MathSearch tool implementation for searching mathematical content
"""

import requests
import json
from typing import Dict, List, Any, Optional

from agent_r1.tool.tool_base import Tool

NUM_RESULTS = 5
DEBUG = True

class MathSearchTool(Tool):
    """
    Tool for searching mathematical content using retrieval API
    """
    def __init__(self, api_url="http://localhost:5100/api/retrieve"):
        """
        Initialize the Math search tool
        
        Args:
            api_url: URL for the retrieval API endpoint
        """
        name = "mathsearch"
        description = "Search for mathematical content based on a query. Useful for finding relevant mathematical proofs and solutions."
        parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A query describing mathematical concepts, theorems, or problems to search for",
                },
            "required": ["query"],
            }
        }
        self.api_url = api_url
        super().__init__(name, description, parameters)
    
    def execute(self, args: Dict) -> str:
        """
        Execute math content search
        
        Args:
            args: Tool parameters, containing:
                - "query": search query string
                - "num_results": optional int to limit number of results
            
        Returns:
            Search results as JSON string
        """
        query = args.get("query")
        if query is None:
            return json.dumps({"error": "Missing required parameter: query"})
            
        num_results = args.get("num_results", NUM_RESULTS)
        
        # Handle both string and list formats for query
        if isinstance(query, str):
            query = [query]  # Convert single string to list
        
        try:
            results = self._search_math_content(query, num_results)
            return self._format_results(results)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        """
        Execute batch search queries
        
        Args:
            args_list: List of query parameter dictionaries
            
        Returns:
            List of search results
        """
        # Collect all queries together
        queries = []
        for args in args_list:
            query = args.get("query", "")
            if isinstance(query, str):
                queries.append(query)
            else:
                queries.extend(query)
        
        # Execute batch query
        if DEBUG:
            print("[DEBUG] Search queries:")
            for query in queries:
                print(query, end="    ")
                
        results = self._search_math_content(queries, num_results=NUM_RESULTS)
        
        # Format and return results
        if "error" in results:
            # If error occurred, return the same error for all queries
            error_message = json.dumps({"error": results["error"]})
            return [error_message] * len(args_list)
        
        # Otherwise format individual results
        formatted_results = []
        for result_group in results.get("results", []):
            formatted_results.append(self._format_results({"results": [result_group]}))
            
        # If we have fewer results than queries, pad with error messages
        while len(formatted_results) < len(args_list):
            formatted_results.append(json.dumps({"error": "No results found"}))
            
        return formatted_results
    
    def _search_math_content(self, query: List[str], num_results = NUM_RESULTS) -> Dict[str, Any]:
        """
        Search for mathematical content using the retrieval API
        
        Args:
            query: List of search keywords
            num_results: Number of results to return
            
        Returns:
            Dictionary containing results or error information
        """
        try:
            payload = {
                'queries': query,
                'num': num_results
            }
            
            response = requests.post(self.api_url, json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to get results, status code: {response.status_code}, response: {response.text}"}
        except Exception as e:
            return {"error": f"API request failed: {str(e)}"}

    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward for search action
        
        Args:
            args: Tool parameters
            result: Tool execution result
            
        Returns:
            Reward value
        """
        # Return neutral reward
        return 0.0

    def _format_results(self, results: Dict[str, Any]) -> str:
        """
        Format search results as a JSON list
        
        Args:
            results: Search results from the API
            
        Returns:
            JSON string containing formatted results
        """
        if "error" in results:
            return json.dumps({"error": results["error"]})
            
        formatted_results = []
        
        # Process the results
        if "results" in results:
            for result_group in results["results"]:
                for item in result_group:
                    # Format each search result
                    formatted_results.append({
                        "text": item.get("text", ""),
                        "score": item.get("score", 0.0)
                    })
                    
        # If no results were found
        if not formatted_results:
            return json.dumps({"error": "No results found"})
                
        return json.dumps(formatted_results, indent=2, ensure_ascii=False)

def test_mathsearch_tool_real_api():
    """
    Test the MathSearchTool functionality using the real API endpoint
    
    This test requires the MathSearch API to be running at the configured URL
    """
    print("Testing MathSearchTool with real API...")
    
    # Create tool instance with default API URL
    tool = MathSearchTool()
    print(f"Using API URL: {tool.api_url}")
    
    try:
        # Test single string query
        print("Testing single query...")
        result = tool.execute({"query": "pythagorean theorem"})
        parsed_result = json.loads(result)
        
        if "error" in parsed_result:
            print(f"API Error: {parsed_result['error']}")
            return
            
        # Print summary of results
        print(f"Received {len(parsed_result)} results")
        if len(parsed_result) > 0:
            print(f"First result: {parsed_result[0]['text'][:100]}...")
            print(f"Score: {parsed_result[0]['score']}")
        
        # Test list query format
        print("\nTesting list query...")
        result = tool.execute({"query": ["euler identity", "calculus integration"]})
        parsed_result = json.loads(result)
        
        if "error" in parsed_result:
            print(f"API Error: {parsed_result['error']}")
        else:
            print(f"Received {len(parsed_result)} results")
        
        # Test with num_results parameter
        print("\nTesting num_results parameter...")
        result = tool.execute({"query": "calculus derivatives", "num_results": 3})
        parsed_result = json.loads(result)
        
        if "error" in parsed_result:
            print(f"API Error: {parsed_result['error']}")
        else:
            print(f"Requested 3 results, received {len(parsed_result)} results")
        
        # Test batch execution
        print("\nTesting batch execution...")
        batch_results = tool.batch_execute([
            {"query": "linear algebra"},
            {"query": "probability theory"}
        ])
        
        print(f"Received {len(batch_results)} batch results")
        for i, result in enumerate(batch_results):
            parsed = json.loads(result)
            if "error" in parsed:
                print(f"Batch {i} error: {parsed['error']}")
            else:
                print(f"Batch {i}: {len(parsed)} results")
        
        print("\nAll MathSearchTool real API tests completed!")
        
    except Exception as e:
        print(f"Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run tests with real API
    test_mathsearch_tool_real_api()
    
    # Alternatively, uncomment to run tests with mocked API
    # test_mathsearch_tool()
