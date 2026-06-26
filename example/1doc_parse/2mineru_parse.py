from src.service.document.load_document import load_document_metadata
from src.service.parse.mineru.api.parser import mineru_parse, mineru_state, parse_doc

username = "administrator"
dataset_name = "leiting"

# Step 2: Use MinuerU to Parse
# Run:  python -m example.1doc_parse.2mineru_parse

# 1.Register new PDFs and load metadata
pdf_files_data = load_document_metadata(username, dataset_name)

# 4.Parse document to JSON (optional, can be done in data_clean step)
for file_id, file_data in pdf_files_data.items():
    parse_doc(file_data)

# 2.Start parsing
# for file_id, file_data in pdf_files_data.items():
#     mineru_parse(file_data)

# 3.Check parsing status (optional)
# for file_id, file_data in pdf_files_data.items():
#     mineru_state(file_data)