#!/usr/bin/env python3
"""
Web UI for A2A Sprint Planning System
A sleek interface to run the sprint automation and view output in real-time
"""
from flask import Flask, render_template, Response, jsonify
import subprocess
import threading
import queue
import time
from datetime import datetime

app = Flask(__name__)

# Global state
process_running = False
output_queue = queue.Queue()
current_process = None

@app.route('/')
def index():
    """Render the main UI page"""
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_process():
    """Start the a2a_agents.py process"""
    global process_running, current_process
    
    if process_running:
        return jsonify({'status': 'error', 'message': 'Process already running'}), 400
    
    # Clear the queue
    while not output_queue.empty():
        output_queue.get()
    
    process_running = True
    
    # Start the process in a background thread
    thread = threading.Thread(target=run_agent_process)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'success', 'message': 'Process started'})

@app.route('/stream')
def stream():
    """Stream output to the client using Server-Sent Events"""
    def generate():
        while True:
            if not output_queue.empty():
                line = output_queue.get()
                yield f"data: {line}\n\n"
            else:
                # Send a heartbeat every 1 second
                time.sleep(0.5)
                if not process_running:
                    yield "data: [DONE]\n\n"
                    break
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/status')
def status():
    """Get current process status"""
    return jsonify({'running': process_running})

def run_agent_process():
    """Run the a2a_agents.py script and capture output"""
    global process_running, current_process
    
    try:
        output_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Starting A2A Sprint Planning System...\n")
        
        # Run the process
        current_process = subprocess.Popen(
            ['python', 'a2a_agents.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output line by line
        for line in iter(current_process.stdout.readline, ''):
            if line:
                output_queue.put(line)
        
        current_process.wait()
        
        if current_process.returncode == 0:
            output_queue.put(f"\n[{datetime.now().strftime('%H:%M:%S')}] ✅ Process completed successfully!\n")
        else:
            output_queue.put(f"\n[{datetime.now().strftime('%H:%M:%S')}] ❌ Process failed with code {current_process.returncode}\n")
    
    except Exception as e:
        output_queue.put(f"\n[{datetime.now().strftime('%H:%M:%S')}] ❌ Error: {str(e)}\n")
    
    finally:
        process_running = False
        current_process = None

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌐 A2A Sprint Planning System - Web UI")
    print("="*70)
    print("\n📱 Open your browser and go to: http://localhost:8000")
    print("\n💡 Press Ctrl+C to stop the server\n")
    
    app.run(debug=False, host='0.0.0.0', port=8000, threaded=True)

