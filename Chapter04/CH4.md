# Chapter 4: Evaluating AI Systems

## Overview

This chapter shows how to evaluate AI systems in practice. We'll cover performance testing, robustness checks, and fairness assessment to ensure your AI works well for everyone.

## Why Evaluate AI Systems?

```mermaid
flowchart TD
    A[AI System] --> B[Performance Check]
    A --> C[Robustness Test]
    A --> D[Fairness Check]
    A --> E[Reliability Test]
    
    B --> F[Does it work accurately?]
    B --> G[Is it fast enough?]
    B --> H[Does it use resources efficiently?]
    
    C --> I[Does it handle edge cases?]
    C --> J[Does it work with noisy inputs?]
    C --> K[Does it handle unexpected data?]
    
    D --> L[Does it treat all users equally?]
    D --> M[Is it free from bias?]
    D --> N[Does it work for different demographics?]
    
    E --> O[Does it work consistently?]
    E --> P[Does it handle errors gracefully?]
    E --> Q[Does it recover from failures?]
```

- **Performance**: Does it work accurately and quickly?
- **Robustness**: Does it handle unexpected inputs well?
- **Fairness**: Does it treat all users equally?
- **Reliability**: Does it work consistently over time?

## Evaluation Framework

```mermaid
flowchart TD
    A[AI System Input] --> B{Evaluation Type}
    
    B --> C[Performance Evaluation]
    B --> D[Robustness Testing]
    B --> E[Fairness Assessment]
    B --> F[Continuous Monitoring]
    
    C --> C1[Accuracy Metrics]
    C --> C2[Speed Metrics]
    C --> C3[Resource Usage]
    
    D --> D1[Input Variations]
    D --> D2[Adversarial Testing]
    D --> D3[Edge Cases]
    
    E --> E1[Demographic Parity]
    E --> E2[Equalized Odds]
    E --> E3[Individual Fairness]
    
    F --> F1[Performance Drift]
    F --> F2[Input Distribution]
    F --> F3[Error Rates]
    
    C1 --> G[Evaluation Results]
    C2 --> G
    C3 --> G
    D1 --> G
    D2 --> G
    D3 --> G
    E1 --> G
    E2 --> G
    E3 --> G
    F1 --> G
    F2 --> G
    F3 --> G
    
    G --> H[Generate Report]
    H --> I[Take Action]
    I --> J[Model Update/Deployment]
```

## 1. Performance Evaluation

Measure how well your model works:

### Accuracy Metrics
- **Precision**: How many positive predictions were actually correct
- **Recall**: How many actual positives were found
- **F1-score**: Balance between precision and recall
- **Accuracy**: Percentage of correct predictions

### Speed Metrics
- **Inference Time**: How fast the model responds
- **Throughput**: Requests handled per second
- **Latency**: Total time including processing

### Performance Evaluation Flow

```mermaid
flowchart LR
    A[Test Dataset] --> B[Model Input]
    B --> C[Model Processing]
    C --> D[Model Output]
    D --> E[Prediction Collection]
    E --> F[Ground Truth Comparison]
    F --> G[Calculate Metrics]
    G --> H[Performance Report]
    
    I[Time Measurement] --> C
    J[Resource Monitoring] --> C
    K[Memory Tracking] --> C
```

## 2. Robustness Testing

Test how your model handles different situations:

### Input Variations
- **Case Changes**: Uppercase, lowercase, mixed case
- **Punctuation**: Different punctuation styles
- **Length**: Very short or very long inputs
- **Format**: Different text structures

### Adversarial Testing
- **Small Changes**: Minor modifications that shouldn't affect output
- **Out-of-Distribution**: Inputs very different from training data
- **Edge Cases**: Unusual or boundary conditions

### Robustness Testing Process

```mermaid
flowchart TD
    A[Original Input] --> B[Apply Transformations]
    B --> C{Transformation Type}
    
    C --> D[Case Changes]
    C --> E[Punctuation Removal]
    C --> F[Character Repetition]
    C --> G[Length Variations]
    C --> H[Special Characters]
    
    D --> I[Modified Input]
    E --> I
    F --> I
    G --> I
    H --> I
    
    I --> J[Model Prediction]
    J --> K[Compare with Original]
    K --> L[Calculate Robustness Score]
    L --> M[Robustness Report]
```

## 3. Fairness Assessment

Ensure your model treats everyone fairly:

### Demographic Parity
- **Gender**: Equal performance across genders
- **Age**: Consistent results across age groups
- **Language**: Works equally well for different languages
- **Culture**: No cultural biases

### Equalized Odds
- **False Positive Rate**: Same rate of incorrect positives across groups
- **False Negative Rate**: Same rate of incorrect negatives across groups
- **Balanced Performance**: No group is disadvantaged

### Fairness Assessment Flow

```mermaid
flowchart TD
    A[Test Dataset] --> B[Group by Demographics]
    B --> C[Gender Groups]
    B --> D[Age Groups]
    B --> E[Language Groups]
    B --> F[Cultural Groups]
    
    C --> G[Calculate Group Metrics]
    D --> G
    E --> G
    F --> G
    
    G --> H[Compare Performance]
    H --> I{Performance Gap?}
    I -->|Yes| J[Identify Bias]
    I -->|No| K[Fair Performance]
    
    J --> L[Bias Mitigation Strategy]
    L --> M[Model Retraining/Adjustment]
    M --> N[Re-evaluate]
    N --> H
```

## Practical Implementation

Here's a simple evaluation system:

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
import time

# Load model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.eval()

# Test data
test_texts = [
    "This product is amazing!",
    "I love this service.",
    "This is terrible quality.",
    "Great experience overall.",
    "Poor customer service."
]

test_labels = [1, 1, 0, 1, 0]  # 1: positive, 0: negative

def evaluate_performance(texts, labels, model, tokenizer):
    """Evaluate model performance"""
    predictions = []
    times = []
    
    for text, label in zip(texts, labels):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1)
        end_time = time.time()
        
        predictions.append(pred.item())
        times.append(end_time - start_time)
    
    accuracy = accuracy_score(labels, predictions)
    avg_time = sum(times) / len(times)
    
    return accuracy, avg_time, predictions

def test_robustness(texts, labels, model, tokenizer):
    """Test model robustness with input variations"""
    print("\n--- Robustness Testing ---")
    
    # Test case sensitivity
    upper_texts = [text.upper() for text in texts[:3]]
    acc_upper, _, _ = evaluate_performance(upper_texts, labels[:3], model, tokenizer)
    print(f"Uppercase accuracy: {acc_upper:.3f}")
    
    # Test with extra spaces
    spaced_texts = [text.replace(" ", "   ") for text in texts[:3]]
    acc_spaced, _, _ = evaluate_performance(spaced_texts, labels[:3], model, tokenizer)
    print(f"Extra spaces accuracy: {acc_spaced:.3f}")

# Run evaluations
print("--- Performance Evaluation ---")
accuracy, avg_time, predictions = evaluate_performance(test_texts, test_labels, model, tokenizer)

print(f"Accuracy: {accuracy:.3f}")
print(f"Average response time: {avg_time:.4f} seconds")
print(f"Throughput: {1/avg_time:.2f} requests/second")

print("\n--- Detailed Report ---")
print(classification_report(test_labels, predictions, target_names=['Negative', 'Positive']))

# Test robustness
test_robustness(test_texts, test_labels, model, tokenizer)
```

**What this code does:**
1. **Loads a model** - BERT for sentiment analysis
2. **Tests performance** - Measures accuracy and speed
3. **Tests robustness** - Checks how it handles input variations
4. **Generates reports** - Provides detailed analysis
5. **Identifies issues** - Finds potential problems

## Best Practices

1. **Test on diverse data** - Include different types of inputs
2. **Compare with baselines** - See how your model performs vs simpler alternatives
3. **Automate testing** - Use scripts for consistent evaluation
4. **Monitor continuously** - Keep tracking performance after deployment

## Key Terms

- **Performance Evaluation**: Measuring accuracy, speed, and efficiency
- **Robustness Testing**: Testing how well models handle variations
- **Fairness Assessment**: Ensuring equal treatment across groups
- **Demographic Parity**: Equal performance across different demographics
- **Equalized Odds**: Similar error rates across groups
- **Adversarial Testing**: Testing with intentionally difficult inputs

## Key Takeaways

1. **Evaluate systematically** - Use structured approaches for testing
2. **Test for robustness** - Make sure it handles edge cases
3. **Check for fairness** - Ensure equal treatment for all users
4. **Monitor continuously** - Keep evaluating after deployment
5. **Automate evaluation** - Make testing consistent and repeatable 