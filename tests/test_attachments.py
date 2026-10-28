import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageDraw
from pypdf import PdfWriter

from attachments import extract_attachment, prepare_image, transcribe_image
from chatbot_core import build_messages, export_markdown


def sample_image():
    image = Image.new("RGB", (700, 250), "white")
    ImageDraw.Draw(image).text((20, 30), "Diffusion problem: D = 0.00001 m^2/s, L = 0.001 m", fill="black", font_size=22)
    output = io.BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


class AttachmentTests(unittest.TestCase):
    @patch("attachments.transcribe_image")
    def test_text_pdf_is_extracted_without_vision_or_api_key(self, transcribe):
        data = (Path(__file__).parent / "fixtures" / "diffusion-problem.pdf").read_bytes()
        passages = extract_attachment(data, "problem.pdf", "")
        self.assertIn("Calculate the flux", passages[0].text)
        self.assertEqual(passages[0].kind, "pdf")
        transcribe.assert_not_called()

    def test_image_validation_and_metadata_free_jpeg(self):
        data = prepare_image(sample_image())
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.mode, "RGB")
        with self.assertRaises(ValueError):
            prepare_image(b"not an image")

    @patch("attachments.Groq")
    def test_vision_uses_image_payload_and_same_provider_key(self, cls):
        provider = cls.return_value.__enter__.return_value
        provider.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="D = 1e-5 m^2/s"))])
        result = transcribe_image(sample_image(), "test-key")
        self.assertIn("D =", result)
        kwargs = provider.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "qwen/qwen3.8-27b")
        self.assertTrue(kwargs["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    @patch("attachments.transcribe_image", return_value="The image contains $D_{AB}=10^{-5}$.")
    def test_image_sources_are_not_mislabeled_as_pdf_pages(self, transcribe):
        passages = extract_attachment(sample_image(), "problem.png", "key")
        self.assertEqual(passages[0].kind, "image")
        system = build_messages("Explain", [], passages, "Explain", "Beginner")[0]["content"]
        self.assertIn("attached image (AI transcription)", system)
        self.assertNotIn("PDF page 1", system)

    @patch("attachments.transcribe_image", return_value="Scanned question: calculate the diffusion flux.")
    def test_scanned_pdf_is_rendered_and_preserves_page_number(self, transcribe):
        buffer = io.BytesIO()
        Image.open(io.BytesIO(sample_image())).save(buffer, format="PDF")
        passages = extract_attachment(buffer.getvalue(), "scan.pdf", "key")
        self.assertEqual(passages[0].kind, "scan")
        self.assertEqual(passages[0].page, 1)
        transcribe.assert_called_once()

    def test_invalid_files_and_scan_limit_are_rejected(self):
        for data, name in [(b"bad", "bad.pdf"), (b"bad", "file.exe"), (b"", "empty.png")]:
            with self.assertRaises(ValueError):
                extract_attachment(data, name, "key")
        buffer = io.BytesIO()
        image = Image.open(io.BytesIO(sample_image()))
        image.save(buffer, format="PDF", save_all=True, append_images=[image] * 5)
        with self.assertRaisesRegex(ValueError, "at most 5"):
            extract_attachment(buffer.getvalue(), "scan.pdf", "key")

    @patch("attachments.transcribe_image")
    def test_blank_pages_do_not_trigger_paid_vision_calls(self, transcribe):
        writer = PdfWriter()
        writer.add_blank_page(100, 100)
        buffer = io.BytesIO()
        writer.write(buffer)
        with self.assertRaisesRegex(ValueError, "no readable"):
            extract_attachment(buffer.getvalue(), "blank.pdf", "key")
        transcribe.assert_not_called()

    def test_export_records_attachment_names_without_image_bytes(self):
        text = export_markdown([{"role": "user", "content": "Read this", "attachments": [{"id": "123", "name": "photo.png"}]}])
        self.assertIn("photo.png", text)
        self.assertNotIn("base64", text)


if __name__ == "__main__":
    unittest.main()
