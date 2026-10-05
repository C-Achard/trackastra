import torch
from trackastra.model import TrackingTransformer
from trackastra.model.predict import predict


def _window(n: int, g: torch.Generator) -> dict:
    return dict(
        coords=torch.rand(n, 2, generator=g) * 200,
        features=torch.randn(n, 3, generator=g),
        pretrained_features=torch.randn(n, 8, generator=g),
        timepoints=torch.randint(0, 4, (n,), generator=g).sort().values,
        labels=torch.arange(1, n + 1),
    )


def test_predict_independent_of_batching():
    """Windows of different lengths are padded when batched; results must not change."""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    model = TrackingTransformer(
        coord_dim=2,
        feat_dim=3,
        pretrained_feat_dim=8,
        reduced_pretrained_feat_dim=4,
        d_model=32,
        window=4,
        causal_norm="none",
    ).eval()
    windows = [_window(n, g) for n in (5, 12, 9)]

    batched = predict(windows, model)
    for i, w in enumerate(windows):
        single = predict([w], model)[0]
        n = len(w["coords"])
        assert torch.allclose(
            torch.as_tensor(batched[i][:n, :n]), torch.as_tensor(single[:n, :n]), atol=1e-5
        ), f"window {i} (n={n}) differs when batched with padding"
