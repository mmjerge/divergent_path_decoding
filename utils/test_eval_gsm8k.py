import pandas as pd
import re
import json
from collections import Counter

def extract_answer_from_response(response):
    """Extract numerical answer from response text."""
    if not response:
        return None
    
    # Primary pattern for the specified format
    primary_pattern = r"The final answer is:\s*\$\\boxed{(\d+)}\$"
    match = re.search(primary_pattern, str(response))
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    
    # Backup patterns for robustness
    patterns = [
        r'\$\\boxed{(\d+)}\$',
        r'####\s*(\d+)',
        r'The answer is:\s*\$\\boxed{(\d+)}\$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, str(response), re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None

def extract_all_answers(item):
    """Extract answers from all sequences and ground truth."""
    answers = {
        'ground_truth': extract_answer_from_response(item.get('correct_answer')),
        'question_index': item.get('question_index')
    }
    
    sequences = item.get('sequences', {})
    for seq_type in ['regular', 'strategic_5th', 'strategic_8th', 'strategic_9th', 'strategic_10th']:
        answers[seq_type] = extract_answer_from_response(sequences.get(seq_type, ''))
    
    return answers

def get_majority_answer(row):
    """Get most common answer across all solutions."""
    answers = [row[col] for col in ['regular', 'strategic_5th', 'strategic_8th', 'strategic_9th', 'strategic_10th'] 
              if pd.notna(row[col])]
    
    if not answers:
        return None
    
    counter = Counter(answers)
    most_common = counter.most_common(1)[0]
    return most_common[0] if most_common[1] > 1 or len(answers) == 1 else None

def calculate_metrics(df, column='regular'):
    """Calculate accuracy metrics for a specific column."""
    valid_answers = df[pd.notna(df[column])]
    total = len(df)
    answered = len(valid_answers)
    correct = sum(valid_answers[column] == valid_answers['ground_truth'])
    
    return {
        'total_questions': total,
        'questions_answered': answered,
        'correct_answers': correct,
        'overall_accuracy': correct / total,
        'accuracy_when_answered': correct / answered if answered > 0 else 0
    }

def analyze_data(data):
    """Analyze all responses and return metrics."""
    results = [extract_all_answers(item) for item in data]
    df = pd.DataFrame(results)
    
    df['majority_answer'] = df.apply(get_majority_answer, axis=1)
    
    metrics = {}
    for column in ['regular', 'strategic_5th', 'strategic_8th', 'strategic_9th', 'strategic_10th']:
        metrics[column] = calculate_metrics(df, column)
    
    metrics['majority_voting'] = calculate_metrics(df, 'majority_answer')
    
    return metrics, df

def main():
    with open('/scratch/mj6ux/Projects/CED/gsm8k_outputs_combined.json', 'r') as f:
        data = json.load(f)
    
    metrics, df = analyze_data(data)
    print("\nResults:")
    for method, results in metrics.items():
        print(f"\n{method.replace('_', ' ').title()}:")
        print(f"Accuracy: {results['overall_accuracy']:.2%}")
        print(f"Correct: {results['correct_answers']}/{results['total_questions']}")
        print(f"Answered: {results['questions_answered']}/{results['total_questions']}")

if __name__ == "__main__":
    main()