import torch
import torch.nn as nn
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from PIL import Image        

class LLaVA(nn.Module):
    def __init__(self):
        super().__init__()
        
        model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            device_map=None,
        ).cuda()
        self.processor = AutoProcessor.from_pretrained(model_id)
        
    def forward(self, images):
        
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe only the motion and gestures of the person in the image focus on hands and face."},
                ],
            },
        ]
                
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        prompts = [prompt for _ in images]

        inputs = self.processor(images=images, text=prompts, return_tensors='pt', padding=True).to(self.model.device, torch.float16)
        outputs = self.model.generate(**inputs, max_new_tokens=256) #pad_token_id=self.processor.tokenizer.eos_token_id
        texts = self.processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        text = [t.split("assistant")[1] for t in texts]

        return text