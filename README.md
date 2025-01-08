# Document & Image Analysis Bot 🤖

A powerful Streamlit application that combines document analysis, image captioning, and voice interaction capabilities using LLMs and modern AI technologies.

## 🌟 Features

- **Document Analysis**: Process and analyze PDF and CSV files
- **Image Captioning**: Generate descriptive captions for images using BLIP model
- **Voice Interaction**: Voice input and text-to-speech output
- **Chat Interface**: Interactive chat with context-aware responses
- **GPU Acceleration**: Utilizes GPU when available for faster processing

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/doc-image-analysis-bot.git
cd doc-image-analysis-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install system dependencies:
   - **Ubuntu/Debian**:
     ```bash
     sudo apt-get install python3-pyaudio portaudio19-dev build-essential python3-dev
     ```
   - **macOS**:
     ```bash
     brew install portaudio
     ```
   - **Windows**: No additional steps needed

4. Run the application:
```bash
streamlit run app.py
```

## 📋 Prerequisites

- Python 3.8+
- Ollama (for LLM support)
- System audio capabilities (for voice features)
- GPU (optional, for faster processing)

## 🛠️ Setup

### Installing Ollama

1. Visit [Ollama's website](https://ollama.ai/)
2. Download and install Ollama for your operating system
3. Pull the Llama2 model:
```bash
ollama pull llama2
```

### GPU Support (Optional)

For NVIDIA GPU support, install PyTorch with CUDA:
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 📁 Project Structure

```
doc-image-analysis-bot/
├── app.py                  # Main application file
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
├── LICENSE                # MIT License
└── README.md              # Project documentation
```

## 🔧 Configuration

The application uses several AI models:
- BLIP (Salesforce) for image captioning
- Llama2 for text processing
- Sentence transformers for document analysis

Models will be downloaded automatically on first use.

## 💡 Usage

1. Launch the application
2. Upload documents (PDF/CSV) or images
3. Use text chat or voice input to interact
4. Explore processed documents and generated captions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⚠️ Known Issues

- Large PDF files might require significant processing time
- Voice input requires a working microphone
- GPU acceleration depends on hardware availability
