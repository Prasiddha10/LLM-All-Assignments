# NLP Assignments Suite - Semester Report

A comprehensive collection of 10 assignments covering the full spectrum of Large Language Models (LLMs), Natural Language Processing (NLP), and multimodal AI applications. This repository demonstrates practical implementations ranging from fundamental NLP preprocessing to advanced multi-agent systems and fine-tuning techniques, organized in progressive learning order.

## 📚 Table of Contents

- [Overview](#overview)
- [Assignment Structure](#assignment-structure)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Individual Assignments](#individual-assignments)
- [Key Learning Outcomes](#key-learning-outcomes)
- [Usage Guidelines](#usage-guidelines)
- [Contributing](#contributing)

## 🎯 Overview

This repository contains 10 comprehensive assignments that explore various aspects of modern AI and NLP, organized in progressive learning order:

### 📊 **Assignment Structure Overview**
```
📁 LLM-ALL-Assignments/
├── 📂 Assignment-1.1-NLP-Preprocessing/           # Fundamental NLP techniques
├── 📂 Assignment-1.2-Word-Embeddings/             # Vector representations
├── 📂 Assignment-1.3-Seq2Seq-Summarization/       # Sequence modeling
├── 📂 Assignment-2.1-Transformer-Finetuning/      # Transformer fine-tuning
├── 📂 Assignment-3.1-RAG-System/                  # Retrieval-Augmented Generation
├── 📂 Assignment-3.2-Agent-and-Multi-Agent-Systems/ # Multi-agent coordination
├── 📂 Assignment-3.3-Fine-Tuning-and-Parameter-Efficient-Methods/ # Parameter efficiency
├── 📂 Assignment-4.0-Prompt-Tuning/               # Prompt engineering
├── 📂 Assignment-5.1-Model-Comparison-Report/     # Comparative analysis
└── 📂 Assignment-5.2-Multimodal-Application-Demo/ # Vision-language applications
```

### 🎓 **Learning Progression**
- **Foundational NLP**: Text preprocessing, word embeddings, and sequence-to-sequence models
- **Transformer Technologies**: Fine-tuning, parameter-efficient methods, and attention mechanisms
- **Advanced Systems**: RAG systems, multi-agent architectures, and prompt engineering
- **Multimodal AI**: Vision-language models and cross-modal applications
- **Comparative Analysis**: Model evaluation and performance benchmarking

## 📁 Assignment Structure

### 🔤 **Natural Language Processing Fundamentals**

#### Assignment 1.1: NLP Preprocessing
- **Tech Stack**: Python, Flask, NLTK, spaCy
- **Features**: Tokenization, lemmatization, stemming, POS tagging, NER
- **Interface**: Web-based demo with RESTful API
- **Key Files**: `app.py`, `templates/index.html`
- **Location**: `Assignment-1.1-NLP-Preprocessing/`

#### Assignment 1.2: Word Embeddings
- **Tech Stack**: Python, Machine Learning libraries
- **Focus**: Word embedding techniques and vector representations
- **Key Files**: `app.py`, `templates/embeddings.html`
- **Location**: `Assignment-1.2-Word-Embeddings/`

#### Assignment 1.3: Seq2Seq Summarization
- **Tech Stack**: TensorFlow, Keras, LSTM, Attention Mechanism
- **Features**: Bidirectional LSTM encoder-decoder with attention
- **Capabilities**: Text summarization with BLEU/ROUGE evaluation
- **Key Files**: `app.py`, `seq2seq_summarization_model.keras`
- **Location**: `Assignment-1.3-Seq2Seq-Summarization/`

### 🔄 **Transformer Architecture & Fine-tuning**

#### Assignment 2.1: Transformer Fine-tuning
- **Tech Stack**: PyTorch, Transformers (HuggingFace), DistilBERT
- **Dataset**: IMDb movie reviews (50K samples)
- **Architecture**: DistilBERT with custom classification head
- **Features**: Comprehensive evaluation, visualization, performance analysis
- **Key Files**: `transformer-finetuning.ipynb`
- **Location**: `Assignment-2.1-Transformer-Finetuning/`

### 🤖 **Advanced AI Systems**

#### Assignment 3.1: RAG System
- **Tech Stack**: Node.js, Express, Python, Chroma DB, Gemini AI
- **Frontend**: Webpack, Chart.js, modern web interface
- **Backend**: Puppeteer web crawling, vector embeddings
- **Features**: Website content extraction, intelligent Q&A, crawl statistics
- **Deployment**: Vercel (frontend), AWS EC2 (backend)
- **Key Files**: `backend/src/index.js`, `frontend/src/scripts.js`
- **Location**: `Assignment-3.1-RAG-System/`

#### Assignment 3.2: Multi-Agent Systems
- **Tech Stack**: Python, Threading, Queue-based messaging
- **Architecture**: 5 specialized agents (Planner, Researcher, Summarizer, Answerer, Coordinator)
- **Features**: Asynchronous processing, thread-safe operations, dynamic task planning
- **Key Files**: `agent-and-multi-agent-systems.ipynb`
- **Location**: `Assignment-3.2-Agent-and-Multi-Agent-Systems/`

#### Assignment 3.3: Parameter-Efficient Fine-tuning
- **Tech Stack**: PyTorch, Transformers, LoRA, Adapters
- **Methods**: Full fine-tuning vs. LoRA vs. Adapter layers
- **Analysis**: Parameter efficiency, training speed, accuracy comparison
- **Visualization**: Comprehensive performance charts and metrics
- **Key Files**: `fine-tuning-parameter-efficient-methods-experiment.ipynb`
- **Location**: `Assignment-3.3-Fine-Tuning-and-Parameter-Efficient-Methods/`

### 💡 **Prompt Engineering & LLM Applications**

#### Assignment 4.0: Prompt Tuning
- **Tech Stack**: Google Gemini API, Python
- **Techniques**: Direct prompts, few-shot learning, chain-of-thought
- **Applications**: Content generation, sentiment analysis
- **Ethics**: Bias, fairness, and privacy considerations
- **Key Files**: `Assignment_4_LLMs.ipynb`
- **Location**: `Assignment-4.0-Prompt-Tuning/`

### 🎨 **Multimodal AI & Model Comparison**

#### Assignment 5.1: Model Comparison Report
- **Focus**: CLIP vs. BLIP comparative analysis
- **Coverage**: Architecture, applications, cross-modal handling
- **Analysis**: Strengths, weaknesses, use-case recommendations
- **Key Files**: `llm_5.1.pdf`
- **Location**: `Assignment-5.1-Model-Comparison-Report/`

#### Assignment 5.2: Multimodal Application Demo
- **Tech Stack**: BLIP, Transformers, Gradio, PIL
- **Application**: Image caption generation
- **Model**: Salesforce BLIP (Vision Transformer + Decoder)
- **Interface**: Interactive Gradio web interface
- **Key Files**: `Image_Caption_Generator_Using_BLIP.ipynb`
- **Location**: `Assignment-5.2-Multimodal-Application-Demo/`

## 🛠️ Technologies Used

### **Programming Languages**
- **Python**: Primary language for ML/AI implementations
- **JavaScript/Node.js**: Web applications and backend services
- **HTML/CSS**: Frontend interfaces

### **AI/ML Frameworks**
- **PyTorch**: Deep learning and transformer models
- **TensorFlow/Keras**: Neural networks and LSTM architectures
- **Transformers (HuggingFace)**: Pre-trained models and fine-tuning
- **NLTK/spaCy**: Natural language processing

### **Specialized Libraries**
- **Chroma DB**: Vector database for embeddings
- **Gradio**: Interactive ML interfaces
- **Chart.js**: Data visualization
- **Puppeteer**: Web scraping and automation

### **APIs & Services**
- **Google Gemini AI**: Language model API
- **Salesforce BLIP**: Multimodal vision-language model

### **Development Tools**
- **Jupyter Notebooks**: Interactive development and experimentation
- **Flask**: Web application framework
- **Webpack**: Frontend bundling and optimization

## 🚀 Setup Instructions

### **Prerequisites**
- Python 3.8+ 
- Node.js 16+
- Git
- Jupyter Notebook/Lab or VS Code

### **General Installation**

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd LLM-ALL-Assignments
   ```

2. **Python Dependencies**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   
   # Install common dependencies
   pip install torch transformers nltk spacy jupyter pandas numpy matplotlib
   python -m spacy download en_core_web_sm
   ```

3. **Node.js Dependencies** (for RAG system)
   ```bash
   cd "Assignment-3.1-RAG-System/backend"
   npm install
   cd "../frontend"
   npm install
   ```

### **API Keys Setup**
Create `.env` files where needed:
```bash
# For Assignment 3.1 (RAG System)
GEMINI_API_KEY=your_gemini_api_key

# For Assignment 4 (Prompt Tuning)
# Add your Gemini API key directly in the notebook
```

## 📖 Individual Assignment Usage

### **Quick Start Guide**

1. **NLP Preprocessing (1.1)**
   ```bash
   cd "Assignment-1.1-NLP-Preprocessing"
   pip install -r requirements.txt
   python app.py
   ```

2. **Transformer Fine-tuning (2.1)**
   ```bash
   cd "Assignment-2.1-Transformer-Finetuning"
   jupyter notebook transformer-finetuning.ipynb
   ```

3. **RAG System (3.1)**
   ```bash
   # Backend
   cd "Assignment-3.1-RAG-System/backend"
   npm start
   
   # Frontend (new terminal)
   cd "../frontend"
   npm run build && npm run start
   ```

4. **Multi-Agent Systems (3.2)**
   ```bash
   cd "Assignment-3.2-Agent-and-Multi-Agent-Systems"
   jupyter notebook agent-and-multi-agent-systems.ipynb
   ```

5. **Parameter-Efficient Fine-tuning (3.3)**
   ```bash
   cd "Assignment-3.3-Fine-Tuning-and-Parameter-Efficient-Methods"
   pip install -r requirements.txt
   jupyter notebook fine-tuning-parameter-efficient-methods-experiment.ipynb
   ```

6. **Image Caption Generator (5.2)**
   ```bash
   cd "Assignment-5.2-Multimodal-Application-Demo"
   pip install -r requirements.txt
   jupyter notebook Image_Caption_Generator_Using_BLIP.ipynb
   ```

## 🎓 Key Learning Outcomes

### **Technical Skills Developed**

1. **NLP Fundamentals**
   - Text preprocessing and feature extraction
   - Word embeddings and semantic representations
   - Sequence-to-sequence modeling with attention

2. **Transformer Architecture**
   - Fine-tuning pre-trained models
   - Parameter-efficient training methods
   - Performance optimization and evaluation

3. **Advanced AI Systems**
   - Retrieval-Augmented Generation (RAG)
   - Multi-agent coordination and communication
   - Vector databases and semantic search

4. **Multimodal AI**
   - Vision-language model integration
   - Cross-modal understanding and generation
   - Interactive AI application development

5. **Practical Implementation**
   - Web application development
   - API design and integration
   - Model deployment and scaling

### **Research & Analysis Skills**

- Comparative model evaluation
- Performance benchmarking
- Ethical AI considerations
- Technical documentation and reporting

## 📋 Usage Guidelines

### **For Students & Researchers**
- Each assignment is self-contained with detailed documentation
- Start with fundamental assignments (1.x) before advanced topics
- Follow setup instructions carefully for each assignment
- Experiment with hyperparameters and model configurations

### **For Practitioners**
- Use as reference implementations for similar projects
- Adapt code for specific use cases and datasets
- Follow best practices demonstrated in each assignment
- Consider ethical implications discussed in the materials

### **For Educators**
- Assignments progress from basic to advanced concepts
- Each includes theoretical background and practical implementation
- Comprehensive evaluation metrics and analysis provided
- Suitable for graduate-level AI/ML courses

## 🤝 Contributing

This repository represents academic work and learning progress. If you find issues or have suggestions:

1. **Issues**: Report bugs or unclear documentation
2. **Improvements**: Suggest enhancements to existing implementations
3. **Extensions**: Propose additional features or experiments
4. **Documentation**: Help improve clarity and completeness

## 📄 License

This repository is for educational purposes. Please respect academic integrity guidelines when using this code for coursework or research.

## 🙋‍♂️ Contact

For questions about specific implementations or concepts covered in these assignments, please refer to the individual assignment documentation or reach out through appropriate academic channels.

---

**Total Assignments**: 10  
**Technologies Covered**: 15+  
**Implementation Types**: Web Apps, Jupyter Notebooks, APIs, Multi-Agent Systems  
**Complexity Level**: Beginner to Advanced  

*This repository demonstrates a complete journey through modern AI and NLP technologies, from foundational concepts to cutting-edge applications.*
