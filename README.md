# tata-rag-system
Multimodal RAG for Tata Motors

1. Domain Identification:
The chosen domain is Automotive Engineering, specifically focusing on Tata Motors vehicle diagnostics, manuals, and technical documentation. Tata Motors produces a wide range of vehicles including passenger cars (Nexon, Tiago), electric vehicles (Nexon EV), and commercial trucks. These vehicles generate extensive multimodal documentation such as service manuals, diagnostic charts, wiring diagrams, and tabular specifications.
   In the context of this assignment, we consider two domain-specific documents: 
 
Nixon Brochure 2026 and 
Nixon EV Brochure.

2. Problem Description:
Automotive engineers, service technicians, and customers often struggle to extract relevant insights from large PDF manuals. These documents contain: 
Textual explanations (fault descriptions, repair steps)
Tables (specifications, torque values, diagnostic codes)
Images (wiring diagrams, component layouts)

Traditional keyword search fails because:
Information is scattered across multiple formats.
Tables and images are not searchable
Technical jargon varies across documents.
For example, a technician may ask:
"What is the battery capacity and charging time of Tata Nexon EV and where is the BMS located?"
This requires retrieving both tabular specs and image-based insights, which traditional systems cannot handle effectively.

3. Why This Problem is Unique?
Unlike generic Q&A systems, automotive documentation includes:- Complex engineering diagrams
Structured tables with critical numerical data
Domain-specific terminology (e.g., ECU, BMS, torque specs)Additionally, safety-critical decisions depend on accurate information retrieval, making precision essential.

4. Why RAG is the Right Approach?
Text and table content are embedded into a vector database, while images are processed using a Vision-Language Model (VLM) to generate descriptive summaries before embedding. This ensures that visual information becomes searchable alongside textual data.
    The core advantage of using a RAG-based approach is its ability to retrieve relevant context dynamically and generate grounded responses using a Large Language Model (LLM). Unlike fine-tuning or static knowledge bases, RAG allows the system to remain flexible and up-to-date with new documents. It also ensures that responses are traceable to specific document sources, improving reliability and transparency. In this use case, the system can answer complex queries such as comparing EV and non-EV features, identifying specifications, or interpreting visual elements within the brochures.
    The system is exposed through a FastAPI-based interface, enabling users to upload documents and query them using natural language. The API supports ingestion, querying, and health monitoring endpoints, ensuring a modular and production-ready design. This architecture separates concerns such as parsing, embedding, retrieval, and response generation, making the system scalable and maintainable.
    A Retrieval-Augmented Generation (RAG) system is ideal because:
    It retrieves relevant chunks from documents instead of relying on memory
    It supports multimodal inputs (text, tables, images)
    It ensures grounded, factual responses with source references
    Compared to fine-tuning:
    RAG is cheaper and more scalable
    Works with continuously updated manuals

    5. Expected Outcomes
    The system should:
    Answer queries from Tata Motors PDFs- Retrieve relevant text, tables, and image summaries- Provide source references (page number, type)
    Example queries:
    “What is the range of Tata Nexon EV?”
    “Show torque specifications table”
    “Explain the wiring diagram of battery system”
    This system will assist:
    Service engineers
    Automotive students
    Vehicle owners
    This assignment demonstrates how a Multimodal RAG system can bridge the gap between unstructured document data and user-friendly information access. By integrating text, tables, and images into a unified retrieval and generation pipeline, the system provides a powerful solution for extracting actionable insights from automotive brochures.

    Problem Statement:
    Multimodal RAG for Tata Motors Vehicle Diagnostics & Documentation
    Customers struggle to extract structured insights like feature, specifications comparisons from automotive brochures that contains:
    Text descriptions
    Technical specification tables
    Images like vehicle parts, diagrams
    A Multimodal RAG System can answer queries like:
    “What is the battery capacity of Nixon EV?”
    “Compare petrol vs EV brochure specifications?”
    “What features are shown in images?”   