from src.service.document.load_document import register_new_pdfs, load_document_metadata

username = "administrator"
dataset_name = "leiting"

# Step 1: Load PDF and Initial Info
# Run:  python -m example.1doc_parse.1load_document

# 1.Load PDF files
register_new_pdfs(username, dataset_name)

pdf_files_data = load_document_metadata(username, dataset_name)
