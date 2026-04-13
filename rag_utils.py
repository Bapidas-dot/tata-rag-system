# Step-1
### Setup the Repo

Start

DEFINE folder SOURCE_DIR for PDFs
DEFINE folder CACHE_DIR for cached data

LOAD embedding_model
LOAD text_llm
LOAD vision_llm

INITIALIZE document_converter
INITIALIZE tokenizer

# Step-2
## Load the Documents

FUNCTION load_cached_documents():

IF cache file exists
    LOAD cached_documents
ELSE
    cached_documents = empty_dictionary

RETURN cached_documents


## Step-3
## Extract PDF into Text, Images, Tables

## Step-3.1

FOR each text_block in document

    chunk_text = extract_text(text_block)
    page_num = get_page_number(text_block)

    ADD {
        content: chunk_text
        metadata: {type: text, page_num: page_num}
    } TO chunks


## Step-3.2 - Extract Images Alone
FOR each Image in document

encoded_image = convert_image_to_base64(image)

image_summary = vision_llm(
    "Analyze this image and summarize it"
     WITH encoded_image
)

ADD {
    content: image_summary
    metadata: {type: image, page_num: page_num}
} TO chunks

## Step-3.3 - Extract Tables Alone
FOR each Table in document

table_markdown = convert_table_to_markdown(table)
page_number = get_page_number(table)

ADD {
    content: table_markdown
    metadata: {type: table, page_num: page_num}
} TO chunks


## Step-4
## Chunk the Files

FUNCTION convert_pdf_to_chunks(pdf_files):

document = parse_pdf(pdf_files)

chunks = empty_list


## Step-5 - Build Vector Data Base and Index
FUNCTION build_vector_database(chunks):

texts = extract_content_from_chunks(chunks)
metadata = extract_metadata_from_chunks(chunks)

embeddings = embedding_model(texts)

index = FAISS_INDEX(embeddings, metadata)

SAVE index

RETURN index








##############################################################################

PART-2 Ingestion Pipeline

#############################################################################

# Step-6 - Ingest PDF Files
FUNCTION ingest_new_pdfs():

cached_docs = load_cached_documents()

FOR each pdf in SOURCE_DIR

    IF pdf not in cached_docs

        chunks = convert_pdf_to_chunks(pdf)

        ADD chunks to vector_database

        cached_docs[pdf] = chunks

SAVE cached_docs
SAVE vector_database

# Step-7 - Retrieve Document
FUNCTION retrieve_documents(query):

index = load_vector_index()

query_embedding = embedding_model(query)

results = index.search(query_embedding, top_k=4)

RETURN results



####################################################################################

# PART-3 Generate Answers

#####################################################################################

# Step-8 - Generate answers
FUNCTION generate_answer(query):

documents = retrieve_documents(query)

context = combine_documents_text(documents)

answer = text_llm(
    PROMPT:
    "Use the context to answer the questions,
    If answer not in context say UNKNOWN"

    CONTEXT = context
    QUESTION = query
)

refences = extract_metadata(documents)

RETURN answer, references


########################################################################

PART-4 Main Program

########################################################################

RUN ingest_new_pdfs()

WHILE user asks question

    query = user_input()

    answer, references = answer_question(query)

    PRINT answer
    PRINT references