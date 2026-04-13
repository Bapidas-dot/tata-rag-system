import fitz  # PyMuPDF


def parse_pdf(file_path):
    """Extract text and images from a PDF file."""
    doc = fitz.open(file_path)
    text_chunks = []
    images = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text:
            text_chunks.append({
                "content": text,
                "page": page_num,
            })

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            images.append({
                "image_bytes": base_image["image"],
                "page": page_num,
            })

    return text_chunks, images
