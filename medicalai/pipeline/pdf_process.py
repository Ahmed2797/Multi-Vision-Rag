import os
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, CompositeElement, Image
from IPython.display import Image 

# -------------------------------
# PDF Loader & Chunker
# -------------------------------
class PDFProcessor:
    def __init__(self, pdf_path, output_path="/figures/"):
        self.pdf_path = pdf_path
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.chunks = []

    def partition_pdf(self):
        self.chunks = partition_pdf(
            filename=self.pdf_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=10000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=6000
        )
        return self.chunks

    def extract_texts_tables(self):
        texts, tables = [], []
        for el in self.chunks:
            if isinstance(el, CompositeElement):
                texts.append(el)
            elif isinstance(el, Table):
                tables.append(el)
        return texts, tables

    def extract_images_base64(self):
        images = []
        for chunk in self.chunks:
            if isinstance(chunk, CompositeElement):
                for el in chunk.metadata.orig_elements:
                    if isinstance(el, Image) and el.metadata.image_base64:
                        images.append(el.metadata.image_base64)
        return images

