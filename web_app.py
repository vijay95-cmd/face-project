from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import sqlite3
import os
import cv2
import numpy as np
from datetime import datetime, timedelta
import json

app = Flask(__name__)
CORS(app)

# Database connection
def get_db():
    conn = sqlite3.connect('face_recognition.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/users')
def get_users():
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM users ORDER BY name')
    users = [dict(row) for row in c.fetchall()]
    conn.close()
    return jsonify(users)

@app.route('/api/attendance/today')
def get_today_attendance():
    conn = get_db()
    c = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    c.execute('''SELECT u.name, a.timestamp, a.confidence
                 FROM attendance a
                 JOIN users u ON a.user_id = u.id
                 WHERE date(a.timestamp) = ?
                 ORDER BY a.timestamp DESC''', (today,))
    records = [dict(row) for row in c.fetchall()]
    conn.close()
    return jsonify(records)

@app.route('/api/attendance/stats')
def get_attendance_stats():
    conn = get_db()
    c = conn.cursor()
    
    # Today's stats
    today = datetime.now().strftime('%Y-%m-%d')
    c.execute('SELECT COUNT(DISTINCT user_id) as users_today FROM attendance WHERE date(timestamp) = ?', (today,))
    today_stats = dict(c.fetchone())
    
    # Total users
    c.execute('SELECT COUNT(*) as total_users FROM users')
    total_users = dict(c.fetchone())
    
    # This week's attendance
    week_start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    c.execute('SELECT COUNT(*) as week_attendance FROM attendance WHERE date(timestamp) >= ?', (week_start,))
    week_stats = dict(c.fetchone())
    
    conn.close()
    
    return jsonify({
        'today_users': today_stats['users_today'],
        'total_users': total_users['total_users'],
        'week_attendance': week_stats['week_attendance']
    })

@app.route('/api/system/status')
def get_system_status():
    status = {
        'model_trained': os.path.exists('trainer.yml'),
        'labels_exist': os.path.exists('labels.npy'),
        'database_exists': os.path.exists('face_recognition.db'),
        'cascade_files': {
            'face': os.path.exists(os.path.join('Haarcascade .xml files', 'haarcascade_frontalface_default.xml')),
            'eye': os.path.exists(os.path.join('Haarcascade .xml files', 'haarcascade_eye.xml'))
        }
    }
    return jsonify(status)

@app.route('/api/attendance/report', methods=['POST'])
def generate_report():
    data = request.json
    report_type = data.get('type', 'daily')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    conn = get_db()
    c = conn.cursor()
    
    if report_type == 'daily':
        date = start_date or datetime.now().strftime('%Y-%m-%d')
        c.execute('''SELECT u.name, a.timestamp, a.confidence
                     FROM attendance a
                     JOIN users u ON a.user_id = u.id
                     WHERE date(a.timestamp) = ?
                     ORDER BY a.timestamp''', (date,))
    else:
        # Weekly or custom range
        start = start_date or (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        end = end_date or datetime.now().strftime('%Y-%m-%d')
        c.execute('''SELECT u.name, date(a.timestamp) as date, COUNT(*) as count
                     FROM attendance a
                     JOIN users u ON a.user_id = u.id
                     WHERE date(a.timestamp) BETWEEN ? AND ?
                     GROUP BY u.name, date(a.timestamp)
                     ORDER BY date(a.timestamp), u.name''', (start, end))
    
    records = [dict(row) for row in c.fetchall()]
    conn.close()
    
    return jsonify(records)

@app.route('/api/user/<int:user_id>/stats')
def get_user_stats(user_id):
    conn = get_db()
    c = conn.cursor()
    
    # Get user info
    c.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = dict(c.fetchone() or {})
    
    if not user:
        conn.close()
        return jsonify({'error': 'User not found'}), 404
    
    # Get attendance stats
    c.execute('''SELECT COUNT(*) as total_recognitions,
                         AVG(confidence) as avg_confidence,
                         MIN(timestamp) as first_seen,
                         MAX(timestamp) as last_seen
                  FROM attendance WHERE user_id = ?''', (user_id,))
    
    stats = dict(c.fetchone())
    
    # Recent activity (last 10 records)
    c.execute('''SELECT timestamp, confidence
                 FROM attendance
                 WHERE user_id = ?
                 ORDER BY timestamp DESC LIMIT 10''', (user_id,))
    
    recent = [dict(row) for row in c.fetchall()]
    
    conn.close()
    
    return jsonify({
        'user': user,
        'stats': stats,
        'recent_activity': recent
    })

if __name__ == '__main__':
    # Create templates directory and basic HTML
    os.makedirs('templates', exist_ok=True)
    
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Recognition System</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-3xl font-bold text-center mb-8">Face Recognition Attendance System</h1>
        
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div class="bg-white rounded-lg shadow-md p-6">
                <h2 class="text-xl font-semibold mb-4">System Status</h2>
                <div class="space-y-2">
                    <div class="flex items-center">
                        <div id="model-status" class="w-3 h-3 bg-gray-400 rounded-full mr-2"></div>
                        <span>Model Trained</span>
                    </div>
                    <div class="flex items-center">
                        <div id="db-status" class="w-3 h-3 bg-gray-400 rounded-full mr-2"></div>
                        <span>Database</span>
                    </div>
                </div>
            </div>
            
            <div class="bg-white rounded-lg shadow-md p-6">
                <h2 class="text-xl font-semibold mb-4">Today's Attendance</h2>
                <div id="today-attendance" class="space-y-2">
                    <p class="text-gray-500">Loading...</p>
                </div>
            </div>
            
            <div class="bg-white rounded-lg shadow-md p-6">
                <h2 class="text-xl font-semibold mb-4">Statistics</h2>
                <div id="stats" class="space-y-2">
                    <p class="text-gray-500">Loading...</p>
                </div>
            </div>
        </div>
        
        <div class="bg-white rounded-lg shadow-md p-6 mt-6">
            <h2 class="text-xl font-semibold mb-4">Registered Users</h2>
            <div id="users-list" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <p class="text-gray-500">Loading...</p>
            </div>
        </div>
    </div>

    <script>
        function renderStatus(data) {
            document.getElementById('model-status').className = data.model_trained ? 'w-3 h-3 bg-green-400 rounded-full mr-2' : 'w-3 h-3 bg-red-400 rounded-full mr-2';
            document.getElementById('db-status').className = data.database_exists ? 'w-3 h-3 bg-green-400 rounded-full mr-2' : 'w-3 h-3 bg-red-400 rounded-full mr-2';
        }

        function loadTodayAttendance() {
            fetch('/api/attendance/today')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('today-attendance');
                    if (!Array.isArray(data) || data.length === 0) {
                        container.innerHTML = '<p class="text-gray-500">No attendance records for today</p>';
                        return;
                    }
                    container.innerHTML = data.map(record => {
                        return `<div class="flex justify-between items-center p-2 bg-gray-50 rounded">
                            <span class="font-medium">${record.name}</span>
                            <span class="text-sm text-gray-600">${record.timestamp}</span>
                        </div>`;
                    }).join('');
                });
        }

        function loadStats() {
            fetch('/api/attendance/stats')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('stats');
                    container.innerHTML = `
                        <div class="flex justify-between">
                            <span>Total Users:</span>
                            <span class="font-bold">${data.total_users || 0}</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Today's Users:</span>
                            <span class="font-bold">${data.today_users || 0}</span>
                        </div>
                        <div class="flex justify-between">
                            <span>This Week:</span>
                            <span class="font-bold">${data.week_attendance || 0}</span>
                        </div>
                    `;
                });
        }

        function loadUsers() {
            fetch('/api/users')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('users-list');
                    if (!Array.isArray(data) || data.length === 0) {
                        container.innerHTML = '<p class="text-gray-500">No users registered yet</p>';
                        return;
                    }
                    container.innerHTML = data.map(user => {
                        return `<div class="p-4 border rounded-lg bg-gray-50">
                            <h3 class="font-semibold">${user.name}</h3>
                            <p class="text-sm text-gray-600">ID: ${user.id}</p>
                        </div>`;
                    }).join('');
                });
        }

        fetch('/api/system/status').then(response => response.json()).then(renderStatus);
        loadTodayAttendance();
        loadStats();
        loadUsers();
        setInterval(loadTodayAttendance, 30000);
    </script>
</body>
</html>
'''
    
    with open('templates/index.html', 'w') as f:
        f.write(html_content)
    
    print('Starting web server on http://localhost:5000')
    app.run(debug=True, host='0.0.0.0', port=5000)
