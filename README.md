# Hallucination-Aware MedAgent (LLaVA-Med + Entropy + PubMed RAG)

Course project exploring hallucination detection and fallback strategies for chest X-ray interpretation using a vision-language model.

## What this does
- Runs LLaVA-Med on a chest X-ray and produces a diagnosis + explanation
- Computes token-level entropy as an uncertainty signal
- If entropy exceeds a threshold, triggers a PubMed retrieval (LangChain RAG) fallback
- Compares performance with/without fallback and analyzes hallucination behavior

## Data
- Kaggle Chest X-ray Pneumonia dataset (NORMAL vs PNEUMONIA)

## How to run
### Google Colab (recommended)
1. Upload `medagent_entropy_rag.ipynb` to Colab
2. Run all cells top-to-bottom

### Local (optional)
1. `pip install -r requirements.txt`
2. Run notebook

## Results (high-level)
- Entropy is a useful warning signal for risky generations
- PubMed RAG did not improve diagnostic accuracy in this setup (future work: CNN fallback)

## Files
- `medagent_entropy_rag.ipynb` — full pipeline + evaluation
