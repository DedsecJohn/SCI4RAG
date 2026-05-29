from src.service.document.load_document import load_document_metadata
from src.service.document.clean_markdown import combine_label_structure

username = "administrator"
dataset_name = "leiting"

# Step 2: Identify Main Sections and Combine Documents
# Run:  python -m example.3data_clean.2combine_label

# 1.Load documents Information
pdf_files_data = load_document_metadata(username, dataset_name)

# 2.Load Markdown and identify chunks main sections
for file_id, file_data in pdf_files_data.items():
    combine_label_structure(file_data)
