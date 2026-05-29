from src.service.parse.delete_parse import delete_parse
from src.service.document.load_document import load_document_metadata

username = "administrator"
dataset_name = "leiting"

# Step 3: Delete parse results
# Run:  python -m example.1doc_parse.3delete_parse

# 1.Load documents Information
pdf_files_data = load_document_metadata(username, dataset_name)

# 2.Delete parse results
for file_id, file_data in pdf_files_data.items():
    delete_parse(file_data)