import os
import base64
from typing import List, Tuple, Any
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import CompositeElement, Table, Image

class PDFProcessor:
    def __init__(self, pdf_path: str, output_path: str = "content_images"):
        self.pdf_path = pdf_path
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.chunks = []

    def partition_pdf(self) -> List[Any]:
        # We keep payload=True so the base64 is available in metadata
        try:
            self.chunks = partition_pdf(
                filename=self.pdf_path,
                strategy="hi_res",
                extract_image_block_types=["Image"],
                extract_image_block_to_payload=True, 
                infer_table_structure=True,
                chunking_strategy="by_title",
                max_characters=10000,
                combine_text_under_n_chars=2000,
                new_after_n_chars=6000
            )
            return self.chunks
        except Exception as e:
            print(f"Partition Error: {e}")
            raise

    def extract_elements(self) -> Tuple[List[str], List[str], List[str]]:
        """A unified extraction method to ensure disk saving matches memory."""
        texts, tables, images_b64 = [], [], []
        image_idx = 0
        
        for el in self.chunks:
            # 1. Logic for Tables
            if isinstance(el, Table):
                tables.append(el.metadata.text_as_html)
            
            # 2. Logic for Texts and Images (nested in Composite or standalone)
            elif isinstance(el, CompositeElement):
                texts.append(el.text)
                # Look for images inside the original elements of the composite chunk
                for sub_el in getattr(el.metadata, "orig_elements", []):
                    if isinstance(sub_el, Image) and sub_el.metadata.image_base64:
                        b64_data = sub_el.metadata.image_base64
                        images_b64.append(b64_data)
                        self._save_to_disk(b64_data, image_idx)
                        image_idx += 1
            
            elif isinstance(el, Image) and el.metadata.image_base64:
                b64_data = el.metadata.image_base64
                images_b64.append(b64_data)
                self._save_to_disk(b64_data, image_idx)
                image_idx += 1
                            
        return texts, tables, images_b64

    def _save_to_disk(self, b64_str: str, index: int):
        """Helper to manually save base64 to the output folder."""
        file_path = os.path.join(self.output_path, f"image_{index}.jpg")
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(b64_str))