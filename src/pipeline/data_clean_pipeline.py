from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.logger import get_user_logger
from src.service.document.load_document import register_new_pdfs, load_document_metadata, parse_path_info
from src.service.parse.mineru.api.parser import mineru_parse, mineru_state
from src.service.extractor.doi import identify_doi_info, identify_DOI, update_doc_info
from src.service.extractor.title import identify_title
from src.service.document.clean_markdown import (
    identify_main_section,
    combile_doc_json,
    combine_label_structure,
    identify_detail,
)
from src.service.extractor.reference import process_references
from src.service.extractor.figure import process_figures
from src.service.extractor.equation import process_equations


class SingleFileCleanPipeline:
    """
    Pipeline for cleaning a single scientific document.

    Executes the full data clean workflow: PDF parsing, DOI identification,
    document structure analysis, and detail extraction (equations, references, figures).
    """

    def __init__(self, file_data: Dict) -> None:
        """
        Initialize single-file clean pipeline.

        Args:
            file_data (dict): The file metadata dictionary loaded from documents.json.
        """
        self.file_data = file_data
        username, dataset_name = parse_path_info(file_data["file_path"])
        self.logger = get_user_logger(username, dataset_name)

    def run(self) -> Dict:
        """
        Execute the full data clean workflow for a single document.

        Returns:
            dict: Result dictionary with keys:
                - file_name (str): The name of the processed file.
                - success (bool): Whether the cleaning completed successfully.
                - error (str or None): Error message if processing failed.
        """
        file_name = self.file_data.get("file_name", "unknown")
        result = {
            "file_name": file_name,
            "success": False,
            "error": None,
        }

        try:
            self.logger.info("Step 1/4: Parsing PDF")
            mineru_parse(self.file_data)
            mineru_state(self.file_data)

            self.logger.info("Step 2/4: Identifying DOI")
            identify_DOI(self.file_data)
            if self.file_data.get("doi"):
                identify_doi_info(self.file_data)
            else:
                identify_title(self.file_data)

            self.logger.info("Step 3/4: Identifying structure")
            identify_main_section(self.file_data)
            identify_detail(self.file_data)
            combine_label_structure(self.file_data)
            combile_doc_json(self.file_data)

            self.logger.info("Step 4/4: Extracting details")
            process_equations(self.file_data)
            process_references(self.file_data)
            process_figures(self.file_data)

            result["success"] = True
            self.logger.success("Processing complete")

        except Exception as e:
            result["error"] = str(e)
            self.logger.exception("Processing failed")

        return result


class BatchDataCleanPipeline:
    """
    Pipeline for cleaning multiple scientific documents in parallel.

    Uses ThreadPoolExecutor to process multiple files concurrently.
    Each file is cleaned independently via SingleFileCleanPipeline.
    """

    def __init__(self, username: str, dataset_name: str, max_workers: int = 4) -> None:
        """
        Initialize batch data clean pipeline.

        Args:
            username (str): The username of the user.
            dataset_name (str): The name of the dataset.
            max_workers (int, optional): Maximum number of parallel workers. Defaults to 4.
        """
        self.username = username
        self.dataset_name = dataset_name
        self.max_workers = max_workers
        self.logger = get_user_logger(username, dataset_name)

    def run(self) -> Dict:
        """
        Execute batch data cleaning for all documents in the dataset.

        Returns:
            dict: Summary dictionary with keys:
                - total (int): Total number of files.
                - success (int): Number of successfully cleaned files.
                - failed (int): Number of failed files.
                - errors (list): List of error dicts for failed files.
        """
        result = {"total": 0, "success": 0, "failed": 0, "errors": []}

        # Register new PDFs, then load all file metadata
        register_new_pdfs(self.username, self.dataset_name)
        pdf_files_data = load_document_metadata(self.username, self.dataset_name)
        result["total"] = len(pdf_files_data)

        if result["total"] == 0:
            self.logger.warning("No files to process")
            return result

        self.logger.info("Processing {total} files (workers={workers})", total=result["total"], workers=self.max_workers)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(SingleFileCleanPipeline(fd).run)
                for fd in pdf_files_data.values()
            ]

            for future in as_completed(futures):
                r = future.result()
                if r["success"]:
                    result["success"] += 1
                else:
                    result["failed"] += 1
                    result["errors"].append(r)

        try:
            update_doc_info(self.username, self.dataset_name)
        except Exception as e:
            result["errors"].append(
                {"file_name": "N/A", "error": f"[BATCH] update_doc_info failed: {str(e)}"}
            )

        self._print_summary(result)
        return result

    def _print_summary(self, result: Dict) -> None:
        """
        Log batch processing summary.

        Args:
            result (dict): The result dictionary from run().
        """
        self.logger.info("Total: {total}  Success: {success}  Failed: {failed}", total=result["total"], success=result["success"], failed=result["failed"])
        for err in result["errors"]:
            self.logger.error("  - {file}: {error}", file=err["file_name"], error=err.get("error", "-"))


if __name__ == "__main__":
    BatchDataCleanPipeline(
        username="administrator", dataset_name="leiting", max_workers=4
    ).run()
