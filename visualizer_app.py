from flask import Flask, render_template, jsonify, request, send_from_directory
import os
import json
import glob

app = Flask(__name__)

# 配置日志根目录
LOG_ROOT = r"e:\CarSim2024.1_Prog\Database\RL_VISION\RL_TC_VISION\logs_TD3"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/folders')
def get_folders():
    """获取所有训练日志文件夹"""
    if not os.path.exists(LOG_ROOT):
        return jsonify([])
    
    folders = [f for f in os.listdir(LOG_ROOT) if os.path.isdir(os.path.join(LOG_ROOT, f))]
    # 按修改时间排序，最新的在前面
    folders.sort(key=lambda x: os.path.getmtime(os.path.join(LOG_ROOT, x)), reverse=True)
    return jsonify(folders)

@app.route('/api/episodes/<folder>')
def get_episodes(folder):
    """获取指定文件夹下的所有episode json文件"""
    folder_path = os.path.join(LOG_ROOT, folder, "episode_data")
    if not os.path.exists(folder_path):
        return jsonify([])
    
    files = glob.glob(os.path.join(folder_path, "episode_*.json"))
    episodes = []
    for f in files:
        filename = os.path.basename(f)
        # 提取episode编号用于排序
        try:
            ep_num = int(filename.replace("episode_", "").replace(".json", ""))
        except:
            ep_num = 0
        episodes.append({"filename": filename, "id": ep_num})
    
    episodes.sort(key=lambda x: x["id"])
    return jsonify(episodes)

@app.route('/api/data/<folder>/<filename>')
def get_episode_data(folder, filename):
    """获取指定episode的详细数据"""
    file_path = os.path.join(LOG_ROOT, folder, "episode_data", filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """处理文件上传并返回解析后的数据"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.json'):
        try:
            content = file.read().decode('utf-8')
            data = json.loads(content)
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": f"Error parsing file: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload a JSON file."}), 400

if __name__ == '__main__':
    print(f"Starting server... Access http://localhost:5000 or http://<your-ip>:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
