# 🖼️ ImgCap1 - AI Image Analysis Tool

A powerful web application that combines computer vision and natural language processing to provide intelligent image analysis. Upload any image and get automated object detection with segmentation masks plus AI-generated captions describing the scene.

## ✨ Features

- **🎯 Object Detection**: Advanced object detection using Mask R-CNN with 80+ COCO object classes
- **🎨 Instance Segmentation**: Visual highlighting of detected objects with colored overlay masks
- **📝 Image Captioning**: AI-generated descriptions using Vision Transformer and GPT-2
- **🖥️ Interactive Web Interface**: Clean, user-friendly Streamlit interface
- **⚡ Real-time Processing**: Fast inference with optimized PyTorch models
- **🎛️ Adjustable Confidence**: Customizable detection confidence threshold

## 🚀 Quick Start

### Prerequisites

- Python 3.13 or higher
- Anaconda/Miniconda (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Code-Junction/ImgCap1.git
   cd ImgCap1
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Access the app**
   - Open your browser and navigate to `http://localhost:8501`
   - Upload an image and click "🔍 Analyze Image"

## 📦 Dependencies

- **Streamlit** (≥1.28.0) - Web framework for the interactive interface
- **PyTorch** (≥2.0.0) - Deep learning framework for model inference
- **Torchvision** (≥0.15.0) - Computer vision models and transforms
- **Transformers** (≥4.30.0) - Hugging Face library for caption generation
- **Pillow** (≥9.5.0) - Image processing and manipulation
- **NumPy** (≥1.24.0) - Numerical computing support

## 🧠 AI Models

### Object Detection & Segmentation
- **Model**: Mask R-CNN with ResNet-50 FPN backbone
- **Dataset**: Pre-trained on MS COCO (80 object classes)
- **Capabilities**: Bounding boxes, instance masks, confidence scores

### Image Captioning
- **Model**: Vision Transformer + GPT-2 (nlpconnect/vit-gpt2-image-captioning)
- **Architecture**: Encoder-decoder with attention mechanisms
- **Output**: Natural language descriptions of image content

## 🎯 Supported Object Classes

The model can detect 80 different object types including:
- **People & Animals**: person, dog, cat, horse, bird, etc.
- **Vehicles**: car, truck, bus, motorcycle, bicycle, etc.
- **Everyday Objects**: chair, table, laptop, phone, book, etc.
- **Food Items**: pizza, banana, apple, sandwich, etc.
- **Sports Equipment**: tennis racket, baseball bat, frisbee, etc.

## 🖥️ Usage

1. **Upload Image**: Use the file uploader to select JPG, JPEG, or PNG images
2. **Adjust Settings**: Use the sidebar to modify detection confidence (0.1-1.0)
3. **Analyze**: Click the "🔍 Analyze Image" button
4. **View Results**:
   - **Caption**: AI-generated description of the image
   - **Object Detection**: Visual representation with detected objects highlighted

## 🔧 Configuration

### Confidence Threshold
- **Range**: 0.1 to 1.0
- **Default**: 0.5
- **Effect**: Higher values show only high-confidence detections

### Performance Optimization
- Models are cached using `@st.cache_resource` for faster subsequent loads
- CPU-optimized inference for broad compatibility
- Warnings suppressed for cleaner output

## 📁 Project Structure

```
ImgCap1/
├── streamlit_app.py      # Main Streamlit application
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── LICENSE              # License information
└── test_performance.py  # Performance testing script
```

## 🛠️ Development

### Local Development
```bash
# Install in development mode
pip install -e .

# Run with debug mode
streamlit run streamlit_app.py --server.runOnSave true
```

### Testing
```bash
python test_performance.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team** for the excellent deep learning framework
- **Hugging Face** for the Transformers library and pre-trained models
- **Streamlit** for the intuitive web framework
- **Microsoft Research** for the COCO dataset
- **Facebook AI Research** for Mask R-CNN architecture

## 📊 Performance

- **Average Processing Time**: 2-5 seconds per image
- **Memory Usage**: ~2-3 GB RAM with models loaded
- **Supported Image Formats**: JPG, JPEG, PNG
- **Max Image Size**: Limited by available system memory

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure stable internet connection for first-time model downloads
   - Check available disk space (models require ~1-2 GB)

2. **Memory Issues**
   - Reduce image resolution for large images
   - Close other applications to free up RAM

3. **Import Errors**
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility (3.13+ recommended)

## 📞 Support

For issues, questions, or contributions, please:
- Open an issue on [GitHub Issues](https://github.com/Code-Junction/ImgCap1/issues)
- Contact the development team

---

**Made with ❤️ by Code-Junction Team**