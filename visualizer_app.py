from flask import Flask, render_template, jsonify, request, send_from_directory
import os
import json
import glob
import subprocess
import atexit
import time

app = Flask(__name__)

# 配置日志根目录
LOG_ROOT = r"e:\CarSim2024.1_Prog\Database\RL_VISION\RL_TC_VISION\logs_TD3_Vision"

# TensorBoard 进程
tensorboard_process = None

def start_tensorboard():
    """启动 TensorBoard 服务"""
    global tensorboard_process
    log_dir = "logs_TD3_Vision"

    # 检查端口 6006 是否已被占用
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('localhost', 6006))
        sock.close()
    except socket.error:
        print("[TensorBoard] 端口 6006 已被占用，TensorBoard 可能已在运行")
        return None

    try:
        # 启动 TensorBoard
        cmd = ['tensorboard', '--logdir=' + log_dir, '--port=6006', '--bind_all']
        tensorboard_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        print(f"[TensorBoard] 启动中... 日志目录: {log_dir}")
        time.sleep(2)  # 等待 TensorBoard 启动

        if tensorboard_process.poll() is None:
            print("[TensorBoard] 启动成功! 访问 http://localhost:6006")
        else:
            stdout, stderr = tensorboard_process.communicate()
            print(f"[TensorBoard] 启动失败: {stderr.decode()}")
            tensorboard_process = None

        return tensorboard_process
    except Exception as e:
        print(f"[TensorBoard] 启动错误: {e}")
        return None

def stop_tensorboard():
    """关闭 TensorBoard 服务"""
    global tensorboard_process
    if tensorboard_process:
        print("\n[TensorBoard] 正在关闭...")
        tensorboard_process.terminate()
        try:
            tensorboard_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tensorboard_process.kill()
        print("[TensorBoard] 已关闭")

# 注册退出时关闭 TensorBoard
atexit.register(stop_tensorboard)

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
    # 自动启动 TensorBoard
    start_tensorboard()

    print(f"Starting server... Access http://localhost:5000 or http://<your-ip>:5000")
    print(f"TensorBoard: http://localhost:6006")
    app.run(debug=True, host='0.0.0.0', port=5000)
