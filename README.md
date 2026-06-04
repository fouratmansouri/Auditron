<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Auditron README</title>
</head>
<body>
  <h1>🤖 Auditron README 🧮</h1>
  <img src="https://i.imgur.com/TTsSThH.jpeg" alt="Auditron Logo" style="width:50%; height:auto;">

  <h2>📋 Overview</h2>
  <p>
    This project, Auditron, aims to develop an AI-powered financial audit assistant (chatbot) to streamline compliance verification and tax calculations for businesses.
    It is part of a semester-long academic project at Esprit School of Engineering, Tunisia, which tackles Agentic AI.
    This tool will combine legal expertise (constitutional and local finance laws) with mathematical precision to automate critical audit tasks, reducing human error and operational costs.
  </p>

  <h2>✨ Features</h2>
  <ul>
    <li><strong>🔍 Intelligent Document Retrieval:</strong> RAG-powered search over Tunisian tax law and regulatory documents for accurate, source-grounded answers.</li>
    <li><strong>🧮 Automated Tax Calculations:</strong> Performs complex fiscal calculations (TVA, IS, IRPP, etc.) based on user-provided financial data.</li>
    <li><strong>⚖️ Compliance Verification:</strong> Cross-checks business operations against current Tunisian tax and accounting regulations.</li>
    <li><strong>🤖 Agentic Workflows:</strong> Multi-step reasoning agents capable of decomposing complex audit tasks and calling external tools autonomously.</li>
    <li><strong>💬 Conversational Interface:</strong> Natural-language chatbot interface usable by non-technical financial staff.</li>
    <li><strong>🌐 Multilingual Support:</strong> Handles queries in French and Arabic, the primary languages of Tunisian business documentation.</li>
    <li><strong>📄 Document Upload & Analysis:</strong> Accepts uploaded financial documents and extracts key data for audit purposes.</li>
    <li><strong>🔒 Explainable Outputs:</strong> Transparent reasoning traces with citations to source laws and regulations.</li>
  </ul>

  <h2>🛠️ Tech Stack</h2>

  <h3>💻 Frontend</h3>
  <p>CSS + HTML</p>

  <h3>⚙️ Backend</h3>
  <p>Flask</p>

  <h3>🧰 Other Tools</h3>
  <ul>
    <li><strong>Ollama:</strong> Local LLM inference runtime for running OLMO2, LLaMA, and DeepSeek models on-premise.</li>
    <li><strong>Docker:</strong> Containerization for reproducible deployment of all services (backend, vector DB, models).</li>
    <li><strong>Git / GitHub:</strong> Version control and collaborative development.</li>
    <li><strong>Postman:</strong> API testing and endpoint documentation.</li>
    <li><strong>Jupyter Notebook:</strong> Prototyping and evaluation of AI components.</li>
  </ul>

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
        ✅ <strong>Advanced Reasoning &amp; Code Generation</strong><br>
        Excels at step-by-step problem-solving in coding and math. Outperforms GPT-4o and Claude-3.5 on LiveCodeBench (65.9% pass@1) and MATH-500 (97.3% pass@1).
      </li>
      <li>
        ✅ <strong>Cost-Efficient Architecture</strong><br>
        Trained under $6M using FP8 precision and multi-token prediction, cutting memory by 75% while preserving accuracy. Distilled from a 671B MoE model with strong performance (1691 CodeForces rating for 32B distilled version).
      </li>
      <li>
        ✅ <strong>Open &amp; Customizable</strong><br>
        Apache 2.0 license for commercial use. Weights and tools available on Hugging Face. Supports fine-tuning via PyTorch/HuggingFace for specialized domains (e.g., code optimization, technical docs).
      </li>
      <li>
        ✅ <strong>Multilingual &amp; Scalable</strong><br>
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
  <img src="https://i.imgur.com/xrF4LHy.png" alt="Retrieval Pipeline Diagram" style="max-width:100%; height:auto;">

  <h3>📄 Document Processing</h3>
  <ul>
    <li><strong>File Formats Supported:</strong> PDF, DOCX, TXT — primarily Tunisian fiscal texts, tax codes, and financial circulars.</li>
    <li><strong>Text Extraction:</strong> Raw text is extracted using <code>PyMuPDF</code> (for PDFs) and <code>python-docx</code> (for Word documents).</li>
    <li><strong>Chunking Strategy:</strong> Documents are split into overlapping chunks (e.g., 512 tokens with 64-token overlap) using LangChain's <code>RecursiveCharacterTextSplitter</code> to preserve contextual continuity.</li>
    <li><strong>Metadata Tagging:</strong> Each chunk is annotated with source filename, article/section number, date of publication, and document type (law, circular, decree).</li>
    <li><strong>Language Detection:</strong> Automatic detection of French vs. Arabic content to route to the appropriate embedding model.</li>
    <li><strong>Preprocessing:</strong> Normalization of Arabic diacritics, removal of headers/footers, and deduplication of repeated legal clauses.</li>
  </ul>

  <h3>💾 Knowledge Base Management</h3>
  <ul>
    <li><strong>Vector Store:</strong> All document embeddings are persisted in a Qdrant collection, organized by document type and year.</li>
    <li><strong>Dual-Embedding Strategy:</strong> French content is embedded with <code>sentence-camembert-large</code>; general/English content uses <code>all-MiniLM-L6-v2</code>. Results are merged at retrieval time.</li>
    <li><strong>Incremental Updates:</strong> New regulatory documents are ingested via a dedicated pipeline that checks for duplicates before upserting into Qdrant.</li>
    <li><strong>Hybrid Search:</strong> Combines dense vector similarity search with BM25 sparse retrieval for improved recall on exact legal terminology.</li>
    <li><strong>Reranking:</strong> A cross-encoder reranker re-scores top-k retrieved chunks before passing them to the LLM context window.</li>
  </ul>

  <h2>💰 Financial Data Integration</h2>

  <h3>📊 Data Sources</h3>
  <p>
    <strong>Jibaya.tn:</strong> The official online portal of the Tunisian tax administration, managed by the Direction Générale des Impôts (DGI). It offers a range of digital services for taxpayers, including online tax declarations, payment of taxes, and access to comprehensive fiscal documentation.
  </p>
  <ul>
    <li><strong>Code Général des Impôts (CGI):</strong> The consolidated Tunisian general tax code, used as the primary legal reference for compliance checks.</li>
    <li><strong>Journal Officiel de la République Tunisienne (JORT):</strong> Official government gazette providing up-to-date legislative texts and decrees.</li>
    <li><strong>BVMT (Bourse des Valeurs Mobilières de Tunis):</strong> Tunisian stock exchange data for publicly listed company financial filings.</li>
    <li><strong>Esprit Financial Department:</strong> Internal financial records and case studies provided by the academic partner for development and evaluation.</li>
  </ul>

  <h3>📈 Market Data APIs</h3>
  <ul>
    <li><strong>BCT (Banque Centrale de Tunisie):</strong> Exchange rates and monetary policy data via the central bank's public data feeds.</li>
    <li><strong>Alpha Vantage / Yahoo Finance:</strong> Used for international market data and currency conversion rates during testing phases.</li>
    <li><strong>Wolfram Alpha API:</strong> Mathematical computation engine integrated as a tool within the LLaMA3.2 agent for complex fiscal formula evaluation.</li>
  </ul>

  <h3>📉 Financial Analytics Tools</h3>
  <ul>
    <li><strong>Pandas &amp; NumPy:</strong> Core data manipulation libraries for processing financial tables and computing indicators.</li>
    <li><strong>Sympy:</strong> Symbolic mathematics library used for algebraic resolution of tax equations (e.g., back-calculating net from gross).</li>
    <li><strong>Matplotlib / Plotly:</strong> Visualization of audit findings, tax liability breakdowns, and trend analysis.</li>
    <li><strong>Custom Tax Engine:</strong> A rules-based Python module encoding Tunisian TVA rates, IRPP brackets, IS rates, and social contribution formulas, callable by the LangGraph agent as a tool.</li>
  </ul>

  <h2>🔒 Security and Compliance</h2>

  <h3>🛡️ LLM Output Safety</h3>
  <ul>
    <li><strong>Hallucination Mitigation:</strong> All LLM-generated financial or legal claims are grounded via RAG — the model is instructed to cite retrieved source chunks and refuse to answer when no relevant document is found.</li>
    <li><strong>Output Validation:</strong> Numerical outputs from tax calculations are cross-validated against the rules-based tax engine before being surfaced to the user.</li>
    <li><strong>Prompt Injection Protection:</strong> User inputs are sanitized and isolated from system instructions; LangChain message templates enforce strict role separation.</li>
    <li><strong>Confidence Scoring:</strong> Low-confidence responses (based on retrieval score thresholds) are flagged with a disclaimer prompting user verification.</li>
  </ul>

  <h3>🔐 Financial Data Protection</h3>
  <ul>
    <li><strong>Local Inference:</strong> All LLMs run on-premise on the DGX A100 server — no financial data is transmitted to external cloud APIs.</li>
    <li><strong>Data Anonymization:</strong> Client financial records used during development are anonymized before ingestion into the knowledge base.</li>
    <li><strong>Access Control:</strong> Role-based access control (RBAC) restricts sensitive document collections and audit logs to authorized users.</li>
    <li><strong>Encryption at Rest:</strong> Qdrant vector collections and uploaded documents are stored on encrypted volumes.</li>
  </ul>

  <h3>⚖️ Regulatory Compliance</h3>
  <ul>
    <li><strong>Tunisian Data Protection Law (Loi n° 2004-63):</strong> Data handling procedures align with the national personal data protection framework.</li>
    <li><strong>Audit Trail:</strong> Every query, retrieved document, and generated response is logged with timestamp and user ID for traceability and regulatory review.</li>
    <li><strong>Disclaimer Mechanism:</strong> The system clearly identifies itself as an AI assistant and reminds users that outputs do not constitute official legal or tax advice.</li>
  </ul>

  <h2>📏 Evaluation Framework</h2>

  <h3>🎯 RAG Quality Metrics</h3>
  <ul>
    <li><strong>Faithfulness:</strong> Measures whether the generated answer is fully supported by the retrieved context (evaluated via RAGAS framework).</li>
    <li><strong>Answer Relevancy:</strong> Assesses how well the response addresses the user's original query.</li>
    <li><strong>Context Recall:</strong> Proportion of ground-truth information successfully retrieved from the knowledge base.</li>
    <li><strong>Context Precision:</strong> Fraction of retrieved chunks that are actually relevant to the query.</li>
    <li><strong>MRR / Hit Rate:</strong> Mean Reciprocal Rank and top-k hit rate to evaluate retrieval ranking quality.</li>
  </ul>

  <h3>📊 Agent Performance Metrics</h3>
  <ul>
    <li><strong>Task Completion Rate:</strong> Percentage of multi-step audit tasks successfully completed end-to-end by the LangGraph agent.</li>
    <li><strong>Tool Call Accuracy:</strong> Rate at which the agent correctly selects and invokes the appropriate tool (tax engine, search, calculator) for a given subtask.</li>
    <li><strong>Latency:</strong> Average end-to-end response time per query, measured separately for retrieval, generation, and tool-call phases.</li>
    <li><strong>Hallucination Rate:</strong> Frequency of factual errors in agent responses, assessed against a manually curated ground-truth dataset of Tunisian tax Q&amp;A pairs.</li>
  </ul>

  <h3>✅ Financial Advice Accuracy</h3>
  <ul>
    <li><strong>Tax Calculation Accuracy:</strong> Numerical outputs validated against ground-truth results computed by certified accountants from Esprit's financial department.</li>
    <li><strong>Legal Citation Accuracy:</strong> Percentage of cited articles correctly matched to the relevant section of the CGI or JORT.</li>
    <li><strong>Human Evaluation:</strong> A panel of finance students and department staff rated response quality on a 5-point Likert scale across dimensions of correctness, clarity, and completeness.</li>
    <li><strong>Regression Testing:</strong> A fixed test suite of 100+ fiscal scenarios is re-run after each model or pipeline update to detect regressions.</li>
  </ul>

  <h2>🚀 Deployment</h2>

  <h3>☁️ Infrastructure</h3>
  <p>
    The project runs on an NVIDIA DGX A100 server equipped with a single 80GB GPU,
    providing high memory bandwidth and compute power optimized for AI workloads.
    This infrastructure enables efficient training and inference for large-scale deep learning models.
  </p>
  <ul>
    <li><strong>OS:</strong> Ubuntu 22.04 LTS</li>
    <li><strong>Containerization:</strong> Docker Compose orchestrates the Flask backend, Qdrant vector database, and Ollama inference server as isolated services.</li>
    <li><strong>Reverse Proxy:</strong> Nginx handles HTTPS termination and request routing to backend services.</li>
    <li><strong>Storage:</strong> NVMe SSD volumes for low-latency vector index access; network-attached storage for document archives.</li>
  </ul>

  <h3>📡 Monitoring</h3>
  <ul>
    <li><strong>Prometheus + Grafana:</strong> Real-time dashboards tracking GPU utilization, inference latency, API request rates, and error rates.</li>
    <li><strong>LangSmith:</strong> LangChain's tracing platform used to monitor agent execution traces, token usage, and chain step durations.</li>
    <li><strong>Qdrant Dashboard:</strong> Built-in Qdrant web UI for monitoring collection health, query performance, and index statistics.</li>
    <li><strong>Structured Logging:</strong> Application logs are structured in JSON and aggregated via a centralized logging service for audit and debugging purposes.</li>
  </ul>

  <h3>📈 Scaling Strategy</h3>
  <ul>
    <li><strong>Horizontal Scaling:</strong> The Flask API is stateless and can be replicated behind a load balancer as query volume grows.</li>
    <li><strong>Model Quantization:</strong> 4-bit and 8-bit quantized versions of LLaMA and DeepSeek are available for lower-memory deployments or edge scenarios.</li>
    <li><strong>Qdrant Distributed Mode:</strong> The vector store can be migrated to a Qdrant cluster for large-scale document ingestion beyond single-node capacity.</li>
    <li><strong>Caching:</strong> Frequently asked questions and their retrieved contexts are cached using Redis to reduce redundant inference costs.</li>
  </ul>

  <h2>👮 Model Governance</h2>

  <h3>🏷️ Versioning</h3>
  <ul>
    <li><strong>Model Registry:</strong> Each LLM version (base model + any fine-tuned adapter) is tagged with a semantic version number and stored alongside its evaluation metrics.</li>
    <li><strong>Pipeline Versioning:</strong> LangChain/LangGraph pipeline configurations are version-controlled in Git; breaking changes require a new major version tag.</li>
    <li><strong>Knowledge Base Snapshots:</strong> Qdrant collection snapshots are taken weekly and before any major document ingestion batch, enabling rollback if retrieval quality degrades.</li>
    <li><strong>Changelog:</strong> A <code>CHANGELOG.md</code> tracks model updates, embedding model changes, and knowledge base additions with date and responsible team member.</li>
  </ul>

  <h3>🧪 Training Data</h3>
  <ul>
    <li><strong>Instruction Dataset:</strong> A curated set of ~500 fiscal Q&amp;A pairs in French, derived from Tunisian tax administration FAQs and accounting textbooks, used for instruction fine-tuning experiments.</li>
    <li><strong>Data Provenance:</strong> All training documents are sourced from public government portals (Jibaya.tn, JORT) or the Esprit financial department with explicit authorization.</li>
    <li><strong>Data Split:</strong> 80% training / 10% validation / 10% test split; test set is held out and never used during model selection or prompt tuning.</li>
    <li><strong>PII Scrubbing:</strong> Automated pipelines remove personal identifiers (names, tax IDs, bank account numbers) before any data enters the training or evaluation pipeline.</li>
  </ul>

  <h3>⚖️ Bias Mitigation</h3>
  <ul>
    <li><strong>Demographic Parity Audit:</strong> Responses are reviewed for disparate treatment across business sizes (SME vs. large enterprise) and sectors (agriculture, services, industry).</li>
    <li><strong>Language Fairness:</strong> Evaluation benchmarks include equal proportions of French and Arabic queries to detect performance gaps across languages.</li>
    <li><strong>Human Review Pipeline:</strong> A sample of 5% of production responses is reviewed weekly by a financial domain expert to catch systematic errors or biases.</li>
    <li><strong>Red-Teaming:</strong> Adversarial prompts (e.g., requests for tax evasion advice) are tested and guarded against via system-level instructions and output filters.</li>
  </ul>

 

  <h2>🚀 Getting Started</h2>
  <ol>
    <li>
      <strong>Clone the repository</strong><br/>
      <code>git clone https://github.com/fouratmansouri/Auditron.git && cd Auditron</code>
    </li>
    <li>
      <strong>Create and activate a virtual environment</strong><br/>
      On macOS/Linux:<br/>
      <code>python3 -m venv venv && source venv/bin/activate</code><br/>
      On Windows:<br/>
      <code>python -m venv venv && .\venv\Scripts\activate</code>
    </li>
    <li>
      <strong>Install Python dependencies</strong><br/>
      <code>pip install -r requirements.txt</code>
    </li>
    <li>
      <strong>Start Qdrant and Ollama via Docker</strong><br/>
      <code>docker-compose up -d qdrant ollama</code>
    </li>
    <li>
      <strong>Pull the required LLM models</strong><br/>
      <code>ollama pull llama3.2 &amp;&amp; ollama pull deepseek-r1:8b &amp;&amp; ollama pull olmo2:13b</code>
    </li>
    <li>
      <strong>Ingest the knowledge base</strong><br/>
      <code>python rag/ingestion/ingest.py --source data/raw/</code>
    </li>
    <li>
      <strong>Run the Flask backend</strong><br/>
      <code>flask --app backend/app.py run --port 5000</code>
    </li>
    <li>
      <strong>Open the frontend</strong><br/>
      Open <code>frontend/index.html</code> in your browser or navigate to <code>http://localhost:5000</code>.
    </li>
  </ol>

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
