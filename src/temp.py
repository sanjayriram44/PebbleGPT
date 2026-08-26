import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from pebblegpt.data.sft_dataset import SFTDataset, load_smoltalk
from pebblegpt.model.modeling import PebbleGPTForCausalLM
from pebblegpt.model.configuration import PebbleGPTConfig

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("loading tokenizer...")
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

print("loading a few SmolTalk conversations...")
convos = load_smoltalk(n_examples=8)
print(f"got {len(convos)} conversations")

ds = SFTDataset(convos, tok, max_len=512)
loader = DataLoader(ds, batch_size=2, shuffle=True)

print("loading annealed checkpoint...")
model = PebbleGPTForCausalLM(PebbleGPTConfig()).to(device)
ckpt = torch.load("checkpoints/anneal_final.pt", map_location=device, weights_only=False)
missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
print(f"missing={missing} unexpected={unexpected}")
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

print("running 5 SFT steps...")
for step, batch in enumerate(loader):
    if step >= 5:
        break
    x = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    out = model(input_ids=x, labels=labels)
    loss = out.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"step {step}: loss={loss.item():.4f}")

print("done")