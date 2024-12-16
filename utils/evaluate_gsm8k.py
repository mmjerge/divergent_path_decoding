import pandas as pd
import re
import json
from IPython.display import display
from collections import Counter
import glob
import os

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

def calculate_majority_accuracy(df):
    """
    Calculate accuracy based on majority voting
    """
    df['majority_answer'] = df.apply(get_majority_answer, axis=1)
    
    total_questions = len(df)
    correct_answers = sum(df['majority_answer'] == df['ground_truth'])
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

def calculate_accuracy(df, column='regular'):
    """
    Calculate accuracy for a specific answer column
    """
    valid_answers = df[pd.notna(df[column])]
    total_questions = len(df)
    questions_with_answers = len(valid_answers)
    correct_answers = sum(valid_answers[column] == valid_answers['ground_truth'])
    
    overall_accuracy = correct_answers / total_questions
    accuracy_when_answered = correct_answers / questions_with_answers if questions_with_answers > 0 else 0
    
    return {
        'total_questions': total_questions,
        'questions_with_answers': questions_with_answers,
        'correct_answers': correct_answers,
        'overall_accuracy': overall_accuracy,
        'accuracy_when_answered': accuracy_when_answered
    }

def extract_answer(text):
    """
    Extract the final numerical answer from a solution text.
    Enhanced to catch more answer formats.
    """
    if not text:
        return None
    
    patterns = [
        r'####\s*(\d+)',  
        r'The answer is\s*(\d+)',
        r'= (\d+)\s*$',
        r'=\s*(\d+)\s*$', 
        r'\$(\d+)\$', 
        r'answer:\s*(\d+)', 
        r'(\d+)\s*is the answer'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return int(matches[-1])
    return None

def extract_all_answers(sequence_text):
    """
    Extract the final answer from each type of sequence response.
    """
    try:
        if isinstance(sequence_text, str):
            sequences = json.loads(sequence_text)
        else:
            sequences = sequence_text
            
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

def extract_answer_from_response(response):
    """
    Extract the numerical answer from a response text.
    Enhanced to catch more answer formats.
    """
    if not response:
        return None
    
    patterns = [
        (r'\$\\boxed{(\d+)}\$', None), 
        (r'\\boxed{(\d+)}', None), 
        (r'answer\s*(?:is|=)?\s*\$?(\d+)\$?', re.IGNORECASE), 
        (r'(?:final|the)\s+answer\s*(?:is|=)?\s*\$?(\d+)\$?', re.IGNORECASE),  # "final/the answer is" formats
        (r'=\s*(\d+)\s*$', None), 
        (r'(?<=\s)(\d+)(?:\s*$|\s*[.!])', None),  
        (r'\$(\d+)\$', None),  # LaTeX style numbers
        (r'≈\s*(\d+)', None),  # Approximately equal
        (r':\s*(\d+)(?:\s*$|\.)', None)  # Number after colon at end
    ]
    
    response = str(response)  # Convert to string to handle any non-string inputs
    
    for pattern, flags in patterns:
        if flags:
            match = re.search(pattern, response, flags)
        else:
            match = re.search(pattern, response)
            
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    
    return None

def create_analysis_dataframe(data):
    """
    Create a pandas DataFrame with the analysis results.
    """
    results = []
    
    for item in data:
        ground_truth = extract_answer(item['correct_answer'])
        answers = extract_all_answers(item['sequences'])
        
        row = {
            'question_index': item['question_index'],
            'ground_truth': ground_truth,
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
    file_pattern = '/scratch/mj6ux/Projects/CED/gsm8k/all/gsm8k_outputs_gpu*_intermediate.json'
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
        
        print("\nRegular (Greedy) Results:")
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
    
    # Combine all dataframes for aggregate analysis
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_majority_metrics = calculate_majority_accuracy(combined_df)
        combined_regular_metrics = calculate_accuracy(combined_df, 'regular')
        
        print("\nAGGREGATE RESULTS ACROSS ALL FILES:")
        
        print("\nRegular (Greedy) Results:")
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