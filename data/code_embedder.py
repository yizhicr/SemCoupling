# data/code_embedder.py
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Optional

class CodeBERTEmbedder:
    def __init__(self, model_name: str = "microsoft/codebert-base", device: Optional[str] = None):
        """
        初始化 CodeBERT 嵌入器
        Args:
            model_name: HuggingFace 模型标识
            device: 运行设备（'cuda', 'cpu'），若为 None 则自动检测
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"加载 CodeBERT 模型到 {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, use_safetensors=True).to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size  # 768

    @torch.no_grad()
    def embed_function(self, code_snippet: str) -> torch.Tensor:
        """
        输入函数源代码字符串，返回语义嵌入向量（768维）
        """
        # 截断过长代码，CodeBERT 最大长度为 512
        inputs = self.tokenizer(
            code_snippet,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        outputs = self.model(**inputs)
        # 取 [CLS] token 的最后一层隐状态作为函数级表示
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)  # [768]
        return cls_embedding.cpu()  # 返回 CPU 张量，便于后续处理