import torch
import transformers
import accelerate

__all__ = ["log_generation"]


def log_generation(
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int,
    step: int,
    accelerator: accelerate.Accelerator,
    top_k: int = 1,
    top_p: float = 0.0,
    min_p: float = 0.0,
    temperature: float = 1.0,
):
    model.eval()
    device_type = next(iter(model.parameters())).device.type
    input_ids = tokenizer("", return_tensors="pt")["input_ids"].to(device_type)
    output = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        temperature=temperature,
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    accelerator.log(
        {
            "generated_text": generated_text,
        },
        step=step,
    )
    return generated_text
