import pandas as pd
import re
import json
from collections import Counter
import glob
import os

def extract_answer_from_response(response):
    """
    Extract the answer from a response text.
    Looks for various answer formats in the response.
    """
    if not response:
        return None

    patterns = [
        r'\$\\boxed{([A-D])}\$',
        r'\$\\boxed{\d+}\$\s*\(([A-D])\)',
        r'final answer is:?\s*\$?\\boxed{([A-D])}\$',
        r'final answer is:?\s*([A-D])[^A-Za-z]',
        r'\(Answer:\s*([A-D])\)',  
        r'answer is\s*[:\s]\s*([A-D])[^A-Za-z]',
        r'(?:answer|option)\s*(?:is|:)?\s*([A-D])[^A-Za-z]'
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, response, re.IGNORECASE)
        last_match = None
        for match in matches:
            last_match = match
        if last_match:
            return last_match.group(1).upper()
    
    return None

def get_majority_answer(row):
    """
    Get the most common answer from all solutions, excluding NaN values
    Returns None if no majority or all values are NaN
    """
    answers = row[['regular', 'strategic_5th', 'strategic_8th', 'strategic_9th', 'strategic_10th']]
    
    valid_answers = [x for x in answers if pd.notna(x)]
    
    if not valid_answers:
        return None
    
    counter = Counter(valid_answers)
    most_common = counter.most_common(1)[0]
    
    if most_common[1] > 1 or len(valid_answers) == 1:
        return most_common[0]
    return None

def calculate_accuracy(df, column='regular'):
    """
    Calculate accuracy for a specific answer column
    """
    valid_answers = df[pd.notna(df[column])]
    total_questions = len(df)
    questions_with_answers = len(valid_answers)
    correct_answers = sum(valid_answers[column] == valid_answers['correct_answer'])
    
    overall_accuracy = correct_answers / total_questions
    accuracy_when_answered = correct_answers / questions_with_answers if questions_with_answers > 0 else 0
    
    return {
        'total_questions': total_questions,
        'questions_with_answers': questions_with_answers,
        'correct_answers': correct_answers,
        'overall_accuracy': overall_accuracy,
        'accuracy_when_answered': accuracy_when_answered
    }

def calculate_majority_accuracy(df):
    """
    Calculate accuracy based on majority voting
    """
    df['majority_answer'] = df.apply(get_majority_answer, axis=1)
    
    total_questions = len(df)
    correct_answers = sum(df['majority_answer'] == df['correct_answer'])
    accuracy = correct_answers / total_questions
    
    questions_with_majority = sum(pd.notna(df['majority_answer']))
    
    accuracy_with_majority = correct_answers / questions_with_majority if questions_with_majority > 0 else 0
    
    return {
        'total_questions': total_questions,
        'questions_with_majority': questions_with_majority,
        'correct_answers': correct_answers,
        'overall_accuracy': accuracy,
        'accuracy_with_majority': accuracy_with_majority
    }

def extract_all_answers(sequences):
    """
    Extract all answers from sequences dictionary
    """
    try:
        return {
            'regular': extract_answer_from_response(sequences.get('regular', '')),
            'strategic_5th': extract_answer_from_response(sequences.get('strategic_5th', '')),
            'strategic_8th': extract_answer_from_response(sequences.get('strategic_8th', '')),
            'strategic_9th': extract_answer_from_response(sequences.get('strategic_9th', '')),
            'strategic_10th': extract_answer_from_response(sequences.get('strategic_10th', ''))
        }
    except:
        return {
            'regular': None,
            'strategic_5th': None,
            'strategic_8th': None,
            'strategic_9th': None,
            'strategic_10th': None
        }

def create_analysis_dataframe(data):
    """
    Create a pandas DataFrame with the analysis results.
    """
    results = []
    
    for item in data:
        # Convert numerical correct_answer to letter (1->B)
        correct_letter = chr(65 + item['correct_answer'])
        answers = extract_all_answers(item['sequences'])
        
        row = {
            'question': item['question'],
            'question_index': item['question_index'],
            'dataset_type': item.get('dataset_type', 'unknown'),
            'correct_answer': correct_letter,
            **answers
        }
        
        results.append(row)
    
    df = pd.DataFrame(results)
    return df

def analyze_file(file_path):
    """
    Analyze a single JSON file and return its metrics
    """
    print(f"\nAnalyzing {os.path.basename(file_path)}:")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    df = create_analysis_dataframe(data)
    majority_metrics = calculate_majority_accuracy(df)
    regular_metrics = calculate_accuracy(df, 'regular')
    
    return {
        'file': os.path.basename(file_path),
        'majority_metrics': majority_metrics,
        'regular_metrics': regular_metrics,
        'dataframe': df
    }

def main():
    # Set pandas display options
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    # Get all intermediate JSON files using absolute path
    file_pattern = '/scratch/mj6ux/Projects/CED/mmlu/all/mmlu_outputs_gpu*_intermediate.json'
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return
    
    # Analyze each file
    results = []
    all_dataframes = []
    
    for file_path in files:
        result = analyze_file(file_path)
        results.append(result)
        all_dataframes.append(result['dataframe'])
        
        print(f"\nResults for {result['file']}:")
        
        print("\nRegular Results:")
        reg_metrics = result['regular_metrics']
        print(f"Total questions: {reg_metrics['total_questions']}")
        print(f"Questions answered: {reg_metrics['questions_with_answers']}")
        print(f"Correct answers: {reg_metrics['correct_answers']}")
        print(f"Overall accuracy: {reg_metrics['overall_accuracy']:.2%}")
        print(f"Accuracy when answered: {reg_metrics['accuracy_when_answered']:.2%}")
        
        print("\nMajority Voting Results:")
        maj_metrics = result['majority_metrics']
        print(f"Total questions: {maj_metrics['total_questions']}")
        print(f"Questions with majority: {maj_metrics['questions_with_majority']}")
        print(f"Correct answers: {maj_metrics['correct_answers']}")
        print(f"Overall accuracy: {maj_metrics['overall_accuracy']:.2%}")
        print(f"Accuracy when majority exists: {maj_metrics['accuracy_with_majority']:.2%}")
    
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_majority_metrics = calculate_majority_accuracy(combined_df)
        combined_regular_metrics = calculate_accuracy(combined_df, 'regular')
        
        print("\nAGGREGATE RESULTS ACROSS ALL FILES:")
        
        print("\nRegular Results:")
        print(f"Total questions: {combined_regular_metrics['total_questions']}")
        print(f"Questions answered: {combined_regular_metrics['questions_with_answers']}")
        print(f"Correct answers: {combined_regular_metrics['correct_answers']}")
        print(f"Overall accuracy: {combined_regular_metrics['overall_accuracy']:.2%}")
        print(f"Accuracy when answered: {combined_regular_metrics['accuracy_when_answered']:.2%}")
        
        print("\nMajority Voting Results:")
        print(f"Total questions: {combined_majority_metrics['total_questions']}")
        print(f"Questions with majority: {combined_majority_metrics['questions_with_majority']}")
        print(f"Correct answers: {combined_majority_metrics['correct_answers']}")
        print(f"Overall accuracy: {combined_majority_metrics['overall_accuracy']:.2%}")
        print(f"Accuracy when majority exists: {combined_majority_metrics['accuracy_with_majority']:.2%}")

if __name__ == "__main__":
    main()