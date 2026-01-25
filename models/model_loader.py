"""
Model loading and initialization for the AI service.
"""
import torch
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def get_device():
    """Get the appropriate device for model loading."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def create_quantization_config():
    """Create quantization configuration for 4-bit loading."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def load_model_and_tokenizer():
    """Load the LLM model and tokenizer with quantization and fallback logic."""
    device = get_device()
    quantization_config = create_quantization_config()
    
    # Try to load Zephyr model with 4-bit quantization first
    try:
        print("Loading Zephyr model with 4-bit quantization...")
        print(f"Current memory usage before loading: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
        print(f"GPU memory before loading: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
        
        model_name = "HuggingFaceH4/zephyr-7b-beta"
        
        print(f"Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded successfully")
        
        print(f"Loading model from {model_name} with quantization...")
        print(f"Quantization config: {quantization_config}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        print(f"Model loaded successfully!")
        print(f"Model device: {model.device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        print(f"Final memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
        print(f"Final GPU memory: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
        print(f"Successfully loaded Zephyr model with 4-bit quantization on {device}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Failed to load Zephyr model: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        
        # Fallback to DialoGPT-medium
        try:
            return _load_fallback_model(quantization_config, "microsoft/DialoGPT-medium")
        except Exception as e2:
            print(f"Failed to load quantized model: {e2}")
            print(f"Error type: {type(e2).__name__}")
            traceback.print_exc()
            
            # Final fallback to CPU with smaller model
            try:
                return _load_cpu_fallback_model()
            except Exception as e3:
                print(f"Failed to load fallback model: {e3}")
                raise Exception("No models could be loaded")


def _load_fallback_model(quantization_config, model_name):
    """Load a fallback model with quantization."""
    print(f"Falling back to {model_name}...")
    
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully")
    
    print(f"Loading model from {model_name} with quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    print(f"Model loaded successfully!")
    print(f"Model device: {model.device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Final memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
    print(f"Final GPU memory: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
    print(f"Successfully loaded {model_name} with 4-bit quantization")
    
    return model, tokenizer


def _load_cpu_fallback_model():
    """Load a smaller model on CPU as final fallback."""
    model_name = "microsoft/DialoGPT-small"
    print(f"Falling back to CPU with {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32
    )
    
    device = "cpu"
    print(f"Model loaded successfully on {device}!")
    print(f"Final memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
    print(f"Successfully loaded DialoGPT-small on {device}")
    
    return model, tokenizer 