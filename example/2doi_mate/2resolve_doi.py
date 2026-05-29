from src.service.document.load_document import load_document_metadata
from src.service.doimeta.fetcher import fetch_DOI_metadata

username = "administrator"
dataset_name = "leiting"

# Step 1: Get Doi and Doi Information
# Run:  python -m example.2doi_mate.2resolve_doi

# 1.Load documents Information
pdf_files_data = load_document_metadata(username, dataset_name)

# 2.Get Scientific Letter Doi
for file_id, file_data in pdf_files_data.items():
    fetch_DOI_metadata(file_data)