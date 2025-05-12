<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
</head>
<body>
  <h1>🤖 Auditron README 🧮</h1>
  <img src="https://i.imgur.com/TTsSThH.jpeg" alt="Auditron Logo" style="width:50%; height:auto;">

  <h2>📋 Overview</h2>
  <p>
    This project, Auditron, aims to develop an AI-powered financial audit assistant (chatbot) to streamline compliance verification and tax calculations for businesses.
    It is part of a semester-long academic project at Esprit School of Engineering, Tunisia, which tackles Agetnic AI.
    This tool will combine legal expertise (constitutional and local finance laws) with mathematical precision to automate critical audit tasks, reducing human error and operational costs.
  </p>

  <h2>✨ Features</h2>

  <h2>🛠️ Tech Stack</h2>
  <h3>💻 Frontend</h3>
  <p>CSS + HTML</p>
  <h3>⚙️ Backend</h3>
  <p>Flask</p>
  <h3>🧰 Other Tools</h3>

  <h2>🧠 AI Components</h2>
  <h3>🔮 Large Language Models</h3>

<section>
  <h4 style="color: #2b6cb0;">For Pre-production: OLMO2:13B</h4>
  <p>
    A state-of-the-art, fully open-source language model by the Allen Institute for AI, designed for transparency and performance.
  </p>
  <ul>
    <li><strong>Fully Open Ecosystem</strong></li>
    <li><strong>High Performance:</strong> Trained on 5T tokens, outperforming Llama-3.1 8B and Qwen 2.5 7B in academic benchmarks.</li>
    <li><strong>Flexible Use:</strong> Supports text generation, reasoning, and fine-tuning; optimized for single-GPU inference.</li>
  </ul>
</section>

<section>
  <h4 style="color: #2b6cb0;">For Production: LLaMA3.2:latest</h4>
  <p>
    A cutting-edge, multimodal model by Meta, optimized for agentic workflows and code-driven applications. Designed for developers building autonomous AI systems and coding tools.
  </p>
  <ul>
    <li><strong>Agentic Task Execution</strong></li>
    <li><strong>Built-in Tool Calling:</strong> Interacts dynamically with external APIs/tools (e.g., Brave Search, Wolfram Alpha) for real-time data, code execution, and query solving.</li>
    <li><strong>Workflow Automation:</strong> Enables multi-step reasoning, parallel tool execution, and iterative problem-solving (e.g., analyzing weather data, synthesizing answers).</li>
  </ul>
</section>

<section>
  <h4 style="color: #2b6cb0;">DeepSeek-R1:8B</h4>
  <p>
    A high-efficiency, open-source model by DeepSeek AI, specialized in reasoning and code generation.
  </p>
  <ul>
    <li>
      ✅ <strong>Advanced Reasoning & Code Generation</strong><br>
      Excels at step-by-step problem-solving in coding and math. Outperforms GPT-4o and Claude-3.5 on LiveCodeBench (65.9% pass@1) and MATH-500 (97.3% pass@1). 
    </li>
    <li>
      ✅ <strong>Cost-Efficient Architecture</strong><br>
      Trained under $6M using FP8 precision and multi-token prediction, cutting memory by 75% while preserving accuracy. Distilled from a 671B MoE model with strong performance (1691 CodeForces rating for 32B distilled version).
    </li>
    <li>
      ✅ <strong>Open & Customizable</strong><br>
      Apache 2.0 license for commercial use. Weights and tools available on Hugging Face. Supports fine-tuning via PyTorch/HuggingFace for specialized domains (e.g., code optimization, technical docs).
    </li>
    <li>
      ✅ <strong>Multilingual & Scalable</strong><br>
      Maintains reasoning in French. Quantized versions support edge deployment on consumer GPUs.
    </li>
    <li>
      ✅ <strong>Transparent Workflow</strong><br>
      Reveals logical steps and thought processes, supporting error tracking and logic auditing.
    </li>
  </ul>
</section>


  <h3>🗄️ Vector Databases</h3>
  <p>Qdrant</p>

  <h3>🔤 Embedding Models</h3>
  <p>dangvantuan/sentence-camembert-large</p>
  <p>sentence-transformers/all-MiniLM-L6-v2</p>

  <h3>🎭 Orchestration Framework</h3>
  <p>Langchain</p>

  <h3>🤖 Agent Framework</h3>
  <p>LangGraph</p>

  <h2>📚 RAG Implementation</h2>
  <h3>🔍 Retrieval Pipeline</h3>
  <img src="https://i.imgur.com/JRAeUOZ.jpeg" alt="Retrieval Pipeline Diagram" style="max-width:100%; height:auto;">

  <h3>📄 Document Processing</h3>
  <h3>💾 Knowledge Base Management</h3>

  <h2>💰 Financial Data Integration</h2>
  <h3>📊 Data Sources</h3>
  <p>
    Jibaya.tn : The official online portal of the Tunisian tax administration, managed by the Direction Générale des Impôts (DGI). It offers a range of digital services for taxpayers, including online tax declarations, payment of taxes, and access to comprehensive fiscal documentation.
  </p>

  <h3>📈 Market Data APIs</h3>
  <h3>📉 Financial Analytics Tools</h3>

  <h2>🔒 Security and Compliance</h2>
  <h3>🛡️ LLM Output Safety</h3>
  <h3>🔐 Financial Data Protection</h3>
  <h3>⚖️ Regulatory Compliance</h3>

  <h2>📏 Evaluation Framework</h2>
  <h3>🎯 RAG Quality Metrics</h3>
  <h3>📊 Agent Performance Metrics</h3>
  <h3>✅ Financial Advice Accuracy</h3>

  <h2>🚀 Deployment</h2>
<h3>☁️ Infrastructure</h3>
<p>
  The project runs on an NVIDIA DGX A100 server equipped with a single 80GB GPU, 
  providing high memory bandwidth and compute power optimized for AI workloads. 
  This infrastructure enables efficient training and inference for large-scale deep learning models.
</p>


  <h3>📡 Monitoring</h3>
  <h3>📈 Scaling Strategy</h3>

  <h2>👮 Model Governance</h2>
  <h3>🏷️ Versioning</h3>
  <h3>🧪 Training Data</h3>
  <h3>⚖️ Bias Mitigation</h3>

  <h2>📁 Directory Structure</h2>

  <h2>🚦 Getting Started</h2>

  <h2>🙏 Acknowledgments</h2>
  <p>This project was developed in partnership with Esprit School of Engineering and Esprit's financial department.</p>
  <p>Special thanks to:</p>
  <ul>
    <li>Prof. Mourad Zerai (<a href="mailto:mourad.zerai@esprit.tn">mourad.zerai@esprit.tn</a>)</li>
    <li>M. Nardine Hanfi (<a href="mailto:nardine.hanfi@esprit.tn">nardine.hanfi@esprit.tn</a>)</li>
    <li>Mr. Souhail Weslati (<a href="mailto:souhail.oueslati@esprit.tn">souhail.oueslati@esprit.tn</a>)</li>
  </ul>
</body>
</html>
