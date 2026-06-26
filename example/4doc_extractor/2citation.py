from src.service.document.load_document import load_document_metadata
from src.service.extractor.citation import process_citations, process_citation_aggregation

username = "administrator"
dataset_name = "leiting"

# Step: Extract in-text citations with context and resolved cited titles
# Run:  python -m example.4doc_extractor.2citation

# 1.Load documents Information
pdf_files_data = load_document_metadata(username, dataset_name)

# 2.Extract citations from document.md and resolve titles via full.md references
for file_id, file_data in pdf_files_data.items():
    process_citations(file_data)

# 3.Supplement cited-article DOIs and aggregate citations per cited article
for file_id, file_data in pdf_files_data.items():
    process_citation_aggregation(file_data)
