import torch
from sky_v1.training.losses import KD3LayerLoss, InfoNCELoss, ReconMSELoss
from sky_v1.training.dpo import dpo_loss

def test_kd3layer_loss_finite_nonnegative():
    b, s, v = 2, 5, 50
    student_logits = torch.randn(b, s, v, requires_grad=True)
    teacher_logits = torch.softmax(torch.randn(5, b, s, v), dim=-1)
    teacher_weights = torch.rand(5).softmax(dim=0)
    labels = torch.randint(0, v, (b, s))
    loss_fn = KD3LayerLoss()
    loss = loss_fn(student_logits, teacher_logits, teacher_weights, labels)
    assert loss.item() >= 0 and torch.isfinite(loss)
    loss.backward()
    assert student_logits.grad is not None and torch.isfinite(student_logits.grad).all()

def test_infonce_loss():
    loss = InfoNCELoss()
    z1 = torch.randn(4, 32)
    z2 = torch.randn(4, 32)
    l = loss(z1, z2)
    assert l.ndim == 0 and l.item() >= 0

def test_recon_mse():
    r = ReconMSELoss()
    a = torch.randn(2, 3, 4)
    b = a + 0.01
    l = r(a, b)
    assert l.item() < 1.0

def test_dpo_loss_basic():
    v = 50
    logits_chosen = torch.randn(2, 8, v, requires_grad=True)
    logits_reject = torch.randn(2, 8, v, requires_grad=True)
    chosen_ids = torch.randint(0, v, (2, 8))
    reject_ids = torch.randint(0, v, (2, 8))
    l = dpo_loss(logits_chosen, logits_reject, chosen_ids, reject_ids)
    assert torch.isfinite(l)
