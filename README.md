# Hallucination-Aware MedAgent (LLaVA-Med + Entropy + PubMed RAG)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/billydavila1010/hallucination-aware-medagent/blob/main/medagent_entropy_rag.ipynb)

Course project evaluating hallucination/uncertainty signals and fallback strategies for chest X-ray reasoning with a vision-language model.

## Disclaimer
For research/educational purposes only. Not a medical device. Do not use for clinical decision-making.

## What this does
- Runs LLaVA-Med on a chest X-ray to produce a pneumonia vs normal **prediction** + explanation
- Computes token-level entropy as an uncertainty signal
- If entropy exceeds a threshold, triggers a PubMed retrieval (LangChain RAG) fallback and re-generates with retrieved context
- Compares performance with/without fallback and analyzes reliability tradeoffs

## Data
- Kaggle Chest X-ray Pneumonia dataset (NORMAL vs PNEUMONIA)

## How to run
### Google Colab (recommended)
1. Open the notebook with the Colab badge above
2. Run all cells top-to-bottom

### Local (optional)
1. `pip install -r requirements.txt`
2. Run the notebook

## Results (high-level)
- Entropy is a useful warning signal for risky generations (threshold used: ~0.65 in experiments)
- PubMed RAG fallback did **not** improve diagnostic accuracy in this setup (~39% with/without RAG)
- Future work: hybrid CNN fallback for high-entropy cases

| Setting | Accuracy | Notes |
|---|---:|---|
| LLaVA-Med baseline | ~0.39 | pneumonia vs normal |
| + PubMed RAG fallback | ~0.39 | no improvement in this setup |

## Files
- `medagent_entropy_rag.ipynb` — full pipeline + evaluation

## References
- LLaVA-Med (model/codebase)
- Kaggle Chest X-ray Pneumonia dataset
- PubMed (retrieval source), LangChain (RAG plumbing)
