from flask import Flask, request, jsonify
from retrieval import retrieve, initialize_resources
import traceback
import argparse

app = Flask(__name__)

@app.route('/api/retrieve', methods=['POST'])
def api_retrieve():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Get queries and num parameters from the request JSON
        queries = data.get('queries', [])
        num = data.get('num', 5)
        
        # Validate input
        if not queries:
            return jsonify({"error": "No queries provided"}), 400
        
        if not isinstance(queries, list):
            queries = [queries]
            
        if not isinstance(num, int) or num <= 0:
            return jsonify({"error": "Invalid num parameter, must be a positive integer"}), 400
            
        # Call the retrieve function from retrieval.py
        results = retrieve(queries, num)
        
        # Format the results to be JSON serializable
        formatted_results = []
        for query_results in results:
            query_formatted = []
            for text, score in query_results:
                query_formatted.append({
                    "text": text,
                    "score": score
                })
            formatted_results.append(query_formatted)
        
        return jsonify({
            "success": True,
            "results": formatted_results
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="启动检索API服务器")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], 
                        help="运行模型的设备 (cuda 或 cpu, 默认: cuda)")
    parser.add_argument("--port", type=int, default=5000, help="服务器端口 (默认: 5000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机 (默认: 0.0.0.0)")
    parser.add_argument("--debug", action="store_true", help="启用debug模式")
    
    args = parser.parse_args()
    
    # 预加载资源
    print(f"预加载模型和索引到 {args.device}...")
    if not initialize_resources(device=args.device):
        print("初始化资源失败，退出")
        exit(1)
    
    print(f"服务器启动在 {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
