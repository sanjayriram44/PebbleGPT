import torch, math
from pebblegpt.model.model import PebbleGPT

model = PebbleGPT()

# 1. Parameter count
print(f"total: {model.num_params()/1e6:.1f}M")
print(f"non-embedding: {model.num_params(non_embedding=True)/1e6:.1f}M")

# 2. Shapes
ids = torch.randint(0, 49152, (2, 128))
logits, loss = model(ids, targets=ids)
assert logits.shape == (2, 128, 49152)

# 3. Initial loss ≈ ln(vocab)
print(f"init loss: {loss.item():.3f}  (expect ~{math.log(49152):.2f})")

# 4. Tying held
assert model.proj_head.weight is model.token_embedding.weight
print("all checks passed")