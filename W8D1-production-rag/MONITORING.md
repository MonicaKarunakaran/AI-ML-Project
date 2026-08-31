# Production RAG Monitoring Strategy

## 1. Objective

The monitoring strategy is designed to detect API failures,
performance degradation, resource problems and deterioration in RAG
answer quality.

---

## 2. Metrics to Monitor

### API Metrics

| Metric | Purpose |
|---|---|
| Request count | Measure traffic |
| Error rate | Detect API failures |
| Response latency | Detect slow requests |
| HTTP status codes | Identify failure types |

### Infrastructure Metrics

| Metric | Purpose |
|---|---|
| CPU utilization | Detect CPU pressure |
| Memory utilization | Detect memory pressure |
| Container restarts | Detect application instability |
| Disk utilization | Prevent storage exhaustion |

### RAG Metrics

| Metric | Purpose |
|---|---|
| Retrieval latency | Identify slow retrieval |
| Retrieval relevance | Measure document quality |
| Context relevance | Check retrieved context |
| Answer correctness | Measure generated answer quality |
| Faithfulness | Detect unsupported answers |
| User feedback | Identify poor responses |

### Model / LLM Metrics

| Metric | Purpose |
|---|---|
| LLM latency | Detect slow model responses |
| Token usage | Monitor resource/cost usage |
| Model errors | Detect model failures |
| Empty/invalid responses | Detect generation problems |

---

## 3. Alert Rules

Initial production alert thresholds:

| Alert | Threshold | Severity |
|---|---:|---|
| API error rate | > 5% for 5 minutes | High |
| API latency | P95 > 3 seconds for 10 minutes | Medium |
| CPU utilization | > 80% for 10 minutes | Medium |
| Memory utilization | > 80% for 10 minutes | High |
| Container restarts | > 3 within 10 minutes | High |
| Retrieval quality | Below evaluation threshold | High |
| RAG faithfulness | Below evaluation threshold | High |

Thresholds should be adjusted after collecting baseline production data.

---

## 4. RAG Quality Evaluation

RAG quality should be evaluated using an evaluation dataset.

Recommended metrics include:

- Context relevance
- Context recall
- Faithfulness
- Answer relevance
- Answer correctness

Ragas can be used for automated RAG evaluation.

MLflow can be used to track evaluation results, model versions,
experiments and performance over time.

---

## 5. Retraining Triggers

Model retraining should not happen automatically for every failure.

Potential triggers include:

1. Persistent degradation in evaluation metrics.
2. Significant increase in incorrect answers.
3. New representative training data becoming available.
4. Domain changes that require model adaptation.
5. Sustained user feedback indicating poor performance.

A retraining workflow should be:

```text
Performance degradation
        ↓
Collect evaluation data
        ↓
Analyze failure cases
        ↓
Retrain / fine-tune model
        ↓
Evaluate new model
        ↓
Compare against current model
        ↓
Deploy only if quality improves