import fitz  # PyMuPDF
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")

class ContextManager:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.contexts = {}
        self.load_contexts()
    
    def load_contexts(self):
        # Chargement processcom.txt
        txt_path = Path("Utils/Data/processcom.txt")
        with open(txt_path, "r", encoding="utf-8") as f:
            self.contexts["processcom"] = {
                "type": "text",
                "content": f.read().split("\n\n")  # Découpage par paragraphes
            }
        
        # Chargement Emotion_Study.pdf
        pdf_path = Path("Utils/Data/Emotion_Study.pdf")
        self.contexts["emotion_study"] = {
            "type": "pdf",
            "content": self._extract_pdf_content(pdf_path)
        }
    
    def _extract_pdf_content(self, pdf_path: Path) -> list:
        """Extrait le contenu structuré du PDF"""
        doc = fitz.open(pdf_path)
        pages_content = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            blocks = [
                f"Page {page_num + 1} - {b[4]}"
                for b in page.get_text("blocks")
                if b[4].strip()
            ]
            pages_content.extend(blocks)
        
        return pages_content
    
    def get_context(self, source: str, question: str, top_k: int = 3) -> str:
        """Récupère le contexte pertinent selon la question"""
        if source not in self.contexts:
            raise ValueError(f"Source {source} inconnue")
        
        content = self.contexts[source]["content"]
        embeddings = self.embedder.encode(content)
        question_embed = self.embedder.encode([question])
        
        similarities = cosine_similarity(question_embed, embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return "\n\n".join([content[i] for i in top_indices])