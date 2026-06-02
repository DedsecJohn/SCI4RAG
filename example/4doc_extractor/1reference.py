from src.service.document.load_document import load_document_metadata
from src.service.extractor.reference import process_references

username = "administrator"
dataset_name = "leiting"

# Step 2: Identify Main Sections and Combine Documents
# Run:  python -m example.4doc_extractor.1reference

# 1.Load documents Information
pdf_files_data = load_document_metadata(username, dataset_name)

# 2.Load Markdown and identify chunks main sections
for file_id, file_data in pdf_files_data.items():
    process_references(file_data)
