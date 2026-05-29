from src.service.document.load_document import load_document_metadata
from src.service.document.clean_markdown import identify_main_section, identify_detail

username = "administrator"
dataset_name = "leiting"

# Step 1: Identify Main Sections and Combine Documents
# Run:  python -m example.3data_clean.1identify_doc_section

# 1.Load documents Information
pdf_files_data = load_document_metadata(username, dataset_name)

# 2.Load Markdown and identify chunks main sections
for file_id, file_data in pdf_files_data.items():
    # if "20260522T173744_536f4abb21" in file_id:
    identify_main_section(file_data)
    identify_detail(file_data)
