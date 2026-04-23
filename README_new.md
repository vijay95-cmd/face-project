# Face Recognition Attendance System

An advanced face recognition system for automated attendance tracking with multiple interfaces and robust features.

## Features

### Core Functionality
- **Real-time Face Recognition**: Live webcam-based recognition with confidence scoring
- **Multi-Algorithm Support**: LBPH, EigenFaces, and FisherFaces recognition algorithms
- **Automated Attendance Tracking**: SQLite database with timestamped records
- **Multiple User Interfaces**: GUI (Tkinter), Web API (Flask), and CLI
- **Dataset Collection**: Automated image capture with quality assessment

### Enhanced Features Added
- **Configuration Management**: YAML-based configuration for easy customization
- **Comprehensive Logging**: Rotating file logs with configurable levels
- **Image Quality Assessment**: Blur detection during dataset collection
- **Model Versioning**: Automatic timestamped model saves
- **Multi-Face Detection**: Simultaneous recognition of multiple people
- **Standardized Thresholds**: Consistent confidence thresholds across modules
- **Robust Error Handling**: Improved exception handling and user feedback
- **Unit Testing**: Basic test suite for core functions

### Interfaces
- **GUI Application**: User-friendly desktop app with voice notifications
- **Web API**: RESTful API for integration with other systems
- **Command Line**: Full CLI control for automation

## Installation

1. Clone the repository
2. Create virtual environment: `python -m venv .venv`
3. Activate: `.venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`

## Configuration

Edit `config.yaml` to customize:
- Recognition parameters (algorithm, thresholds)
- File paths
- Logging settings
- Web server configuration

## Usage

### Quick Start
1. Collect dataset: `python dataset.py`
2. Train model: `python train.py`
3. Start recognition: `python recognition.py`

### Advanced Usage
- GUI: `python gui_app.py`
- Web API: `python web_app.py`
- CLI Menu: `python main_launcher.py`

## Project Structure

```
face-project/
├── config.yaml          # Configuration file
├── requirements.txt     # Python dependencies
├── config.py           # Configuration loader
├── logger.py           # Logging setup
├── dataset.py          # Dataset collection with quality check
├── train.py            # Model training
├── recognition.py      # Real-time recognition with attendance
├── gui_app.py          # Desktop GUI application
├── web_app.py          # Flask web API
├── attendance_system.py # Analytics and reporting
├── main_launcher.py    # CLI menu system
├── simple_launcher.py  # Simplified launcher
├── tests.py            # Unit tests
├── trainer.yml         # Trained model (generated)
├── labels.npy          # Label mappings (generated)
├── face_recognition.db # SQLite database (generated)
└── dataset/            # Training images directory
```

## API Endpoints

- `GET /` - Homepage
- `GET /api/users` - List all users
- `GET /api/today_attendance` - Today's attendance
- `GET /api/attendance_stats` - Statistics
- `GET /api/system_status` - Health check
- `GET /api/report` - Generate reports

## Contributing

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Use logging for debugging

## License

MIT License